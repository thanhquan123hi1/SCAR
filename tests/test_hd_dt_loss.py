"""Unit tests for hierarchical batch Dice and distance-transform Hausdorff surrogate losses."""
from __future__ import annotations

import pytest
import torch

from training.loss.losses import BOUNDARY_DT_MODES, SegmentationLoss
from training.train import parse_args


def test_zero_weights_exact_backward_compatibility():
    torch.manual_seed(42)
    logits = torch.randn(4, 4, 32, 32, requires_grad=True)
    target = torch.randint(0, 4, (4, 32, 32))

    base = SegmentationLoss(
        n_classes=4,
        ce_weight=0.2,
        dice_weight=0.8,
        dice_class_weights=[0.5, 1.0, 1.0, 1.0],
        negative_slice_handling="batch",
        include_background=True,
        dice_normalization="classes",
    )
    extended = SegmentationLoss(
        n_classes=4,
        ce_weight=0.2,
        dice_weight=0.8,
        hier_dice_weight=0.0,
        boundary_dt_weight=0.0,
        dice_class_weights=[0.5, 1.0, 1.0, 1.0],
        negative_slice_handling="batch",
        include_background=True,
        dice_normalization="classes",
    )
    out_base = base(logits, target)
    out_ext = extended(logits, target)
    assert set(out_base.keys()) == set(out_ext.keys())
    for k in out_base:
        assert torch.allclose(out_base[k], out_ext[k], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("mode", BOUNDARY_DT_MODES)
@pytest.mark.parametrize("target_kind", ["normal", "all_empty", "all_scar", "partial_empty"])
def test_hd_dt_loss_finite_and_grads_edge_cases(mode, target_kind):
    torch.manual_seed(7)
    logits = torch.randn(4, 4, 64, 64, requires_grad=True)
    if target_kind == "all_empty":
        target = torch.zeros(4, 64, 64, dtype=torch.long)
    elif target_kind == "all_scar":
        target = torch.full((4, 64, 64), 3, dtype=torch.long)
    elif target_kind == "partial_empty":
        target = torch.zeros(4, 64, 64, dtype=torch.long)
        target[0, 20:36, 20:36] = 1
        target[0, 24:30, 24:30] = 2
        target[1, 25:35, 25:35] = 3
    else:
        target = torch.zeros(4, 64, 64, dtype=torch.long)
        target[:, 16:48, 16:48] = 1
        target[:, 22:40, 22:40] = 2
        target[:, 26:34, 26:34] = 3

    crit = SegmentationLoss(
        n_classes=4,
        ce_weight=0.2,
        dice_weight=0.8,
        hier_dice_weight=0.20,
        boundary_dt_weight=0.10,
        boundary_dt_mode=mode,
        dice_class_weights=[0.5, 1.0, 1.0, 1.0],
        negative_slice_handling="batch",
        include_background=True,
        dice_normalization="classes",
    )
    out = crit(logits, target)
    for k, v in out.items():
        assert torch.isfinite(v), f"{mode}/{target_kind}: non-finite {k}={v}"
        assert v.item() >= 0.0, f"{mode}/{target_kind}: negative {k}={v}"
    out["loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("mode", BOUNDARY_DT_MODES)
def test_distant_fp_penalized_more_than_near_fp(mode):
    target = torch.zeros(1, 64, 64, dtype=torch.long)
    target[0, 28:36, 28:36] = 3  # central scar block

    def make_logits(fp_r, fp_c):
        logits = torch.full((1, 4, 64, 64), -3.0)
        logits[0, 0] = 3.0
        logits[0, 0, 28:36, 28:36] = -3.0
        logits[0, 3, 28:36, 28:36] = 3.0
        # Inject identical 4x4 FP scar blob at (fp_r, fp_c)
        logits[0, 0, fp_r:fp_r + 4, fp_c:fp_c + 4] = -3.0
        logits[0, 3, fp_r:fp_r + 4, fp_c:fp_c + 4] = 3.0
        return logits

    crit = SegmentationLoss(
        n_classes=4,
        ce_weight=0.2,
        dice_weight=0.8,
        hier_dice_weight=0.2,
        boundary_dt_weight=0.1,
        boundary_dt_mode=mode,
        dice_class_weights=[0.5, 1.0, 1.0, 1.0],
        negative_slice_handling="batch",
        include_background=True,
        dice_normalization="classes",
    )
    near = crit(make_logits(36, 30), target)["boundary_dt_loss"].item()
    far = crit(make_logits(2, 2), target)["boundary_dt_loss"].item()
    assert far > near * 5.0, f"{mode}: expected far ({far}) >> near ({near})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bf16 check")
@pytest.mark.parametrize("mode", BOUNDARY_DT_MODES)
def test_bf16_cuda_grads_finite(mode):
    logits = torch.randn(4, 4, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.zeros(4, 64, 64, device="cuda", dtype=torch.long)
    target[0, 10:30, 10:30] = 1
    target[0, 15:25, 15:25] = 2
    target[1, 20:30, 20:30] = 3
    # slices 2 & 3 intentionally empty
    crit = SegmentationLoss(
        n_classes=4,
        ce_weight=0.2,
        dice_weight=0.8,
        hier_dice_weight=0.20,
        boundary_dt_weight=0.10,
        boundary_dt_mode=mode,
        dice_class_weights=[0.5, 1.0, 1.0, 1.0],
        negative_slice_handling="batch",
        include_background=True,
        dice_normalization="classes",
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = crit(logits, target)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "cfg_path,expected_mode",
    [
        ("training/config/models/lightweight_e5_b1r_s3_l1_symdt_bce.yaml", "sym_bce"),
        ("training/config/models/lightweight_e5_b1r_s3_l2_karimi_hddt.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b1r_s3_l3_strong_karimi.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b1r_s3_l4_hybrid_farfp.yaml", "hybrid_karimi_farfp"),
        ("training/config/models/lightweight_e5_b2_s3_anat_mean_l2_karimi.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b2_s3_anat_mean_ring_ctx_l4.yaml", "hybrid_karimi_farfp"),
        ("training/config/models/lightweight_e5_b1r_s3_l5_calib_tversky.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b1r_s3_l6_calib_farfp.yaml", "hybrid_karimi_farfp"),
        ("training/config/models/lightweight_e5_b1r_s3_l7_scale_karimi.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b2sa_s3_anat_mean_l5.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b1r_s3_l8_mild_scale_karimi.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b2sa_s3_anat_mean_ring_ctx_l5.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b1r_s3_l9_slice_hier_karimi.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b2sa_s3_anat_mean_l8.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b2gsa_s3_anat_mean_ring_ctx_l9.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b1r_s3_l10_fullrange_karimi.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b2_s3_anat_mean_l10.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b1r_s3_ring_ctx_l10.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b1r_s3_anat_mean_l5.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b2_s3_anat_mean_l5.yaml", "karimi_mse"),
        ("training/config/models/lightweight_e5_b1r_s3_l11_fullreach_aar_karimi.yaml", "karimi_mse"),
    ],
)
def test_loss_trial_yamls_parse_cleanly(cfg_path, expected_mode):
    args, merged = parse_args(["--config", cfg_path, "--dataset", "myopspp_unet2d_roi128_1mm"])
    assert args.boundary_dt_mode == expected_mode
    assert args.hier_dice_weight > 0.0
    assert args.boundary_dt_weight > 0.0
    assert args.max_epochs == 300
    assert args.batch_size == 16
    assert args.seed == 1234
    assert args.no_val is True


def test_ring_tversky_and_deadzone_margin_grads():
    torch.manual_seed(99)
    logits = torch.randn(4, 4, 64, 64, requires_grad=True)
    target = torch.zeros(4, 64, 64, dtype=torch.long)
    target[0, 16:48, 16:48] = 1
    target[0, 24:36, 24:36] = 2
    target[1, 20:40, 20:40] = 3
    crit = SegmentationLoss(
        n_classes=4,
        ce_weight=0.2,
        dice_weight=0.8,
        hier_dice_weight=0.25,
        ring_tversky_beta=0.60,
        aar_tversky_beta=0.40,
        boundary_dt_weight=0.25,
        boundary_dt_mode="hybrid_karimi_farfp",
        boundary_dt_margin=4.0,
        boundary_dt_empty_val=1.5,
        dice_class_weights=[0.5, 1.0, 1.0, 1.0],
        negative_slice_handling="batch",
        include_background=True,
        dice_normalization="classes",
    )
    out = crit(logits, target)
    assert torch.isfinite(out["loss"]) and out["loss"].item() > 0.0
    out["loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_per_region_scales_and_weights_bf16_and_empty():
    torch.manual_seed(123)
    logits = torch.randn(4, 4, 64, 64, requires_grad=True)
    target = torch.zeros(4, 64, 64, dtype=torch.long)
    target[0, 16:48, 16:48] = 1
    target[0, 22:34, 22:34] = 2
    target[0, 26:30, 26:30] = 3
    # Slices 1..3 are completely empty (0) to verify zero-GT stability
    crit = SegmentationLoss(
        n_classes=4,
        ce_weight=0.40,
        dice_weight=0.60,
        ce_class_weights=[0.80, 1.10, 0.80, 1.15],
        dice_class_weights=[0.5, 1.15, 0.75, 1.1],
        hier_dice_weight=0.30,
        hier_region_weights=[0.35, 0.40, 0.25],
        ring_tversky_beta=0.64,
        aar_tversky_beta=0.39,
        boundary_dt_weight=0.30,
        boundary_dt_mode="karimi_mse",
        boundary_dt_power=2.0,
        boundary_dt_scale=12.0,
        boundary_dt_scales=[5.0, 10.0, 14.0],
        boundary_dt_region_weights=[0.35, 0.45, 0.20],
        boundary_dt_max_steps=32,
        boundary_dt_empty_val=2.0,
        boundary_dt_max_norm=2.5,
        negative_slice_handling="batch",
        include_background=True,
        dice_normalization="classes",
    )
    out = crit(logits.to(torch.bfloat16), target)
    assert torch.isfinite(out["loss"]) and 0.0 < out["loss"].item() < 20.0
    out["loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="boundary_dt_scales"):
        SegmentationLoss(boundary_dt_scales=[5.0, -1.0, 10.0])
    with pytest.raises(ValueError, match="boundary_dt_max_norm"):
        SegmentationLoss(boundary_dt_max_norm=0.0)


def test_slice_hier_and_slice_karimi_bf16_and_empty():
    torch.manual_seed(77)
    logits = torch.randn(4, 4, 64, 64, requires_grad=True)
    target = torch.zeros(4, 64, 64, dtype=torch.long)
    target[0, 10:50, 10:50] = 1
    target[0, 20:35, 20:35] = 2
    target[0, 24:30, 24:30] = 3
    # Small apical slice on index 1, completely empty slices on 2 and 3
    target[1, 28:36, 28:36] = 1
    target[1, 30:34, 30:34] = 3
    crit = SegmentationLoss(
        n_classes=4,
        ce_weight=0.40,
        dice_weight=0.60,
        ce_class_weights=[0.85, 1.05, 0.85, 1.15],
        dice_class_weights=[0.5, 1.1, 0.80, 1.1],
        hier_dice_weight=0.20,
        slice_hier_weight=0.15,
        ring_tversky_beta=0.62,
        aar_tversky_beta=0.42,
        boundary_dt_weight=0.25,
        boundary_dt_slice_weight=0.50,
        boundary_dt_mode="karimi_mse",
        boundary_dt_scales=[8.0, 13.0, 15.0],
        boundary_dt_empty_val=1.75,
        negative_slice_handling="batch",
        include_background=True,
        dice_normalization="classes",
    )
    out = crit(logits.to(torch.bfloat16), target)
    assert torch.isfinite(out["loss"]) and 0.0 < out["loss"].item() < 15.0
    out["loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="boundary_dt_slice_weight"):
        SegmentationLoss(boundary_dt_slice_weight=1.5)


def test_myo_fp_alpha_and_norm_myo_dt_loss():
    torch.manual_seed(99)
    target = torch.zeros(2, 64, 64, dtype=torch.long)
    target[0, 12:52, 12:52] = 1
    target[0, 20:28, 20:28] = 2

    # Logits predicting normal myocardium (class 1) everywhere in myocardium, except a false Edema blob inside GT normal myo
    logits_fp_in_myo = torch.zeros(2, 4, 64, 64)
    logits_fp_in_myo[:, 0] = 3.0
    logits_fp_in_myo[0, 1, 12:52, 12:52] = 5.0
    logits_fp_in_myo[0, 2, 20:28, 20:28] = 5.0
    logits_fp_in_myo[0, 2, 40:46, 40:46] = 6.0  # false Edema inside GT normal myocardium

    base_crit = SegmentationLoss(
        n_classes=4, ce_weight=0.40, dice_weight=0.60, hier_dice_weight=0.25,
        boundary_dt_weight=0.25, boundary_dt_mode="karimi_mse", boundary_dt_scales=[8.5, 14.0, 16.0],
        boundary_dt_myo_fp_alpha=0.0, negative_slice_handling="batch", include_background=True, dice_normalization="classes",
    )
    boost_crit = SegmentationLoss(
        n_classes=4, ce_weight=0.40, dice_weight=0.60, hier_dice_weight=0.25,
        boundary_dt_weight=0.25, boundary_dt_mode="karimi_mse", boundary_dt_scales=[8.5, 14.0, 16.0],
        boundary_dt_myo_fp_alpha=1.5, negative_slice_handling="batch", include_background=True, dice_normalization="classes",
    )
    out_base = base_crit(logits_fp_in_myo, target)
    out_boost = boost_crit(logits_fp_in_myo, target)
    assert out_boost["boundary_dt_loss"].item() > out_base["boundary_dt_loss"].item()

    logits_grad = logits_fp_in_myo.clone().detach().requires_grad_(True)
    nm_crit = SegmentationLoss(
        n_classes=4, ce_weight=0.40, dice_weight=0.60, hier_dice_weight=0.25,
        boundary_dt_weight=0.25, boundary_dt_mode="karimi_mse", boundary_dt_scales=[8.5, 15.0, 16.0],
        norm_myo_dt_weight=0.12, negative_slice_handling="batch", include_background=True, dice_normalization="classes",
    )
    out_nm = nm_crit(logits_grad.to(torch.bfloat16), target)
    assert "norm_myo_dt_loss" in out_nm and torch.isfinite(out_nm["norm_myo_dt_loss"])
    assert torch.isfinite(out_nm["loss"]) and out_nm["loss"].item() > 0.0
    out_nm["loss"].backward()
    assert logits_grad.grad is not None and torch.isfinite(logits_grad.grad).all()

    with pytest.raises(ValueError, match="accum_steps=1"):
        SegmentationLoss(norm_myo_dt_weight=0.1).validate_accumulation(2)
    with pytest.raises(ValueError, match="boundary_dt_myo_fp_alpha"):
        SegmentationLoss(boundary_dt_myo_fp_alpha=-0.5)
    with pytest.raises(ValueError, match="norm_myo_dt_weight"):
        SegmentationLoss(norm_myo_dt_weight=-0.1)


def test_karimi_qlog_empty_vals_and_intra_myo_far_fp():
    torch.manual_seed(123)
    target = torch.zeros(2, 64, 64, dtype=torch.long)
    target[0, 8:56, 8:56] = 1
    target[0, 12:20, 12:20] = 2  # GT Edema top-left

    # High-confidence Far-FP in Normal Myocardium on opposite wall (44:52, 44:52), distance > 20 px
    logits = torch.full((2, 4, 64, 64), -3.0)
    logits[:, 0] = 4.0
    logits[0, 1, 8:56, 8:56] = 4.0
    logits[0, 2, 12:20, 12:20] = 4.0
    logits[0, 2, 44:52, 44:52] = 6.5  # p_edema ~ 0.92 inside target == 1
    logits.requires_grad_(True)

    crit_mse = SegmentationLoss(
        ce_weight=0.0, dice_weight=0.0, boundary_dt_weight=0.25, boundary_dt_mode="karimi_mse",
        boundary_dt_scales=[9.0, 15.5, 16.0], boundary_dt_empty_vals=[1.5, 1.5, 2.5],
        negative_slice_handling="batch", include_background=True, dice_normalization="classes",
    )
    out_mse = crit_mse(logits, target)
    out_mse["loss"].backward()
    grad_mse_far = logits.grad[0, 2, 44:52, 44:52].abs().mean().item()

    logits.grad = None
    crit_qlog = SegmentationLoss(
        ce_weight=0.0, dice_weight=0.0, boundary_dt_weight=0.25, boundary_dt_mode="karimi_qlog",
        boundary_dt_scales=[9.0, 15.5, 16.0], boundary_dt_empty_vals=[1.5, 1.5, 2.5],
        negative_slice_handling="batch", include_background=True, dice_normalization="classes",
    )
    out_qlog = crit_qlog(logits, target)
    out_qlog["loss"].backward()
    grad_qlog_far = logits.grad[0, 2, 44:52, 44:52].abs().mean().item()
    assert grad_qlog_far > 3.0 * grad_mse_far

    # bf16 stability check
    logits_bf = logits.detach().clone().requires_grad_(True)
    out_bf = crit_qlog(logits_bf.to(torch.bfloat16), target)
    assert torch.isfinite(out_bf["loss"])
    out_bf["loss"].backward()
    assert torch.isfinite(logits_bf.grad).all()

    # intra_myo_far_fp_loss: zero gradient outside myocardium (target == 0), strong gradient
    # at far intra-myo AAR FP (>14 px), and boundary stability (no 1/p_ring explosion)
    logits_intra = logits.detach().clone().requires_grad_(True)
    crit_intra = SegmentationLoss(
        ce_weight=0.0, dice_weight=0.0, intra_myo_far_fp_weight=0.15,
        negative_slice_handling="batch", include_background=True, dice_normalization="classes",
    )
    out_intra = crit_intra(logits_intra, target)
    assert "intra_myo_far_fp_loss" in out_intra and out_intra["intra_myo_far_fp_loss"].item() > 0.0
    out_intra["loss"].backward()
    # Strictly zero gradient on GT background (target == 0)
    torch.testing.assert_close(
        logits_intra.grad[:, :, 0:6, 0:6],
        torch.zeros_like(logits_intra.grad[:, :, 0:6, 0:6]),
        atol=1e-7,
        rtol=0,
    )
    assert logits_intra.grad[0, 2, 44:52, 44:52].abs().mean().item() > 0.0
    # Nearby myocardium (within 14 px of GT AAR) gets zero AAR-far penalty on class 2
    assert logits_intra.grad[0, 2, 20:24, 20:24].abs().max().item() < 1e-5
    # Conditional ratio preservation inside target==1: p_0 and p_1 receive gradients strictly
    # proportional to their softmax probabilities, preventing boundary spikes when p_ring is tiny
    logits_edge = torch.full((1, 4, 32, 32), -8.0)
    logits_edge[:, 0] = 4.0  # p_0 ~ 0.98
    t_edge = torch.zeros(1, 32, 32, dtype=torch.long)
    t_edge[0, 8:24, 8:24] = 1
    t_edge[0, 10:14, 10:14] = 2
    logits_edge[0, 1, 20:22, 20:22] = -3.0  # tiny p_1 ~ 0.001
    logits_edge[0, 2, 20:22, 20:22] = 0.0   # p_2 ~ 0.018
    logits_edge.requires_grad_(True)
    crit_intra(logits_edge, t_edge)["loss"].backward()
    assert logits_edge.grad[0, 2, 20:22, 20:22].abs().max().item() < 1e-4

    # empty_slice_fp_weight: penalizes Scar FP inside GT Edema on empty-Scar slice
    # while leaving positive-Scar slices untouched on empty_slice_fp_loss
    t_empty = torch.zeros(2, 64, 64, dtype=torch.long)
    t_empty[0, 8:56, 8:56] = 1
    t_empty[0, 12:24, 12:24] = 2  # Case3003 z=2 pattern: GT Edema present, GT Scar == 0
    t_empty[1, 8:56, 8:56] = 1
    t_empty[1, 12:24, 12:24] = 3  # Slice 1 HAS GT Scar (never penalized by empty_slice_fp_loss)
    log_empty = torch.full((2, 4, 64, 64), -3.0)
    log_empty[:, 0] = 4.0
    log_empty[0, 3, 12:24, 12:24] = 6.0  # False Scar inside GT Edema on empty-Scar slice
    log_empty[1, 3, 40:48, 40:48] = 6.0  # False Scar on positive-Scar slice
    log_empty.requires_grad_(True)
    crit_empty = SegmentationLoss(
        ce_weight=0.0, dice_weight=0.0, empty_slice_fp_weight=0.12,
        negative_slice_handling="batch", include_background=True, dice_normalization="classes",
    )
    out_empty = crit_empty(log_empty, t_empty)
    assert "empty_slice_fp_loss" in out_empty and out_empty["empty_slice_fp_loss"].item() > 0.0
    out_empty["loss"].backward()
    # Slice 0 empty Scar inside GT Edema gets pushed down on class 3 (stronger inside ring than outside)
    assert log_empty.grad[0, 3, 12:24, 12:24].mean().item() > 1e-5
    # Slice 1 (has_scar=True AND has_aar=True) gets strictly zero gradient from empty_slice_fp_loss
    torch.testing.assert_close(
        log_empty.grad[1],
        torch.zeros_like(log_empty.grad[1]),
        atol=1e-7,
        rtol=0,
    )

    with pytest.raises(ValueError, match="accum_steps=1"):
        SegmentationLoss(intra_myo_far_fp_weight=0.1).validate_accumulation(2)
    with pytest.raises(ValueError, match="accum_steps=1"):
        SegmentationLoss(empty_slice_fp_weight=0.1).validate_accumulation(2)
    with pytest.raises(ValueError, match="boundary_dt_empty_vals"):
        SegmentationLoss(boundary_dt_empty_vals=[1.5, -0.1, 2.0])
    with pytest.raises(ValueError, match="intra_myo_far_fp_weight"):
        SegmentationLoss(intra_myo_far_fp_weight=-0.2)
    with pytest.raises(ValueError, match="empty_slice_fp_weight"):
        SegmentationLoss(empty_slice_fp_weight=-0.2)





