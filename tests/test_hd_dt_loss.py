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




