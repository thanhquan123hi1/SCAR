"""Adversarial Stress Test Harness for Milestone 4 (R4)
SCAR Cardiac MRI Segmentation Codebase Refactoring

Empirical Challenger 1 Test Suite:
1. Extreme Loss & Gradient Stress Tests (1 vs 100k imbalance, pure background, pure foreground, extreme +-100 logits, backward pass NaNs/Infs).
2. Anatomical Postprocessing Stress Tests (apical scar through-plane connection, floating scars, transmural vs subendocardial, patchy scars, 2D/3D tensor/numpy).
3. Metric Calibration & Nature Methods 2024 Alignment Tests (Symmetric TN, Dynamic FOV HD95, single-voxel, 2D vs 3D scar volume/mass).
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.loss import (
    MultiClassSoftDiceLoss,
    OneVsRestCompoundLoss,
    FocalTverskyLoss,
    ce_dice_loss,
    build_loss,
    masks_to_one_vs_rest,
    LOSS_REGISTRY,
)
from training.metrics import (
    compute_fov_diagonal,
    hd95_binary,
    dice_score,
    dice_score_symmetric,
    iou_score,
    iou_score_symmetric,
    mean_dice,
    calculate_scar_metrics,
    ScarMetrics,
)
from training.postprocess.anatomical import (
    enforce_anatomical_constraints,
    postprocess_predictions,
)


# =============================================================================
# CATEGORY 1: LOSS FUNCTIONS & GRADIENT BACKPROPAGATION STRESS TESTS
# =============================================================================

def test_loss_extreme_imbalance_1_vs_100k():
    """Stress test: 1 foreground voxel vs 100,000 background voxels."""
    num_classes = 5
    B, H, W = 1, 316, 317  # 100,172 voxels
    
    labels = torch.zeros((B, H, W), dtype=torch.long)
    labels[0, 150, 150] = 3  # exactly 1 scar voxel
    
    # 1. MultiClassSoftDiceLoss
    dice_loss = MultiClassSoftDiceLoss(num_classes=num_classes, smooth=1.0)
    logits_mc = torch.randn((B, num_classes, H, W), requires_grad=True)
    l_dice = dice_loss(logits_mc, labels)
    assert torch.isfinite(l_dice), "MultiClassSoftDiceLoss non-finite on 1:100k imbalance"
    assert l_dice.item() >= 0.0, "Negative loss in MultiClassSoftDiceLoss"
    l_dice.backward()
    assert logits_mc.grad is not None
    assert torch.isfinite(logits_mc.grad).all(), "MultiClassSoftDiceLoss grad non-finite on 1:100k imbalance"

    # 2. OneVsRestCompoundLoss
    ovr_loss = OneVsRestCompoundLoss(
        num_classes=num_classes,
        bce_weight=0.5,
        dice_weight=0.5,
        focal_weight=1.0,
        focal_gamma=2.0,
        pos_weight=[2.0, 3.0, 10.0, 2.0],
        class_weights=[1.0, 1.0, 4.0, 1.0],
    )
    logits_ovr = torch.randn((B, num_classes - 1, H, W), requires_grad=True)
    l_ovr = ovr_loss(logits_ovr, labels)
    assert torch.isfinite(l_ovr), "OneVsRestCompoundLoss non-finite on 1:100k imbalance"
    assert l_ovr.item() >= 0.0
    l_ovr.backward()
    assert logits_ovr.grad is not None
    assert torch.isfinite(logits_ovr.grad).all(), "OneVsRestCompoundLoss grad non-finite on 1:100k imbalance"

    # 3. FocalTverskyLoss
    ft_loss = FocalTverskyLoss(num_classes=num_classes, alpha=0.3, beta=0.7, gamma=1.33)
    logits_ft = torch.randn((B, num_classes, H, W), requires_grad=True)
    l_ft = ft_loss(logits_ft, labels)
    assert torch.isfinite(l_ft), "FocalTverskyLoss non-finite on 1:100k imbalance"
    assert l_ft.item() >= 0.0
    l_ft.backward()
    assert logits_ft.grad is not None
    assert torch.isfinite(logits_ft.grad).all(), "FocalTverskyLoss grad non-finite on 1:100k imbalance"

    # 4. ce_dice_loss
    logits_ce = torch.randn((B, num_classes, H, W), requires_grad=True)
    l_ce = ce_dice_loss(logits_ce, labels, num_classes=num_classes, class_weights=[0.1, 1.0, 1.0, 5.0, 1.0])
    assert torch.isfinite(l_ce), "ce_dice_loss non-finite on 1:100k imbalance"
    l_ce.backward()
    assert logits_ce.grad is not None
    assert torch.isfinite(logits_ce.grad).all(), "ce_dice_loss grad non-finite on 1:100k imbalance"


def test_loss_pure_background_slices():
    """Stress test: pure background slices (y=0 everywhere) across 2D and 3D."""
    num_classes = 5
    
    # Batch with mixed 2D slices
    B, H, W = 4, 64, 64
    labels_2d = torch.zeros((B, H, W), dtype=torch.long)
    
    # Confident correct background prediction (logits[0] = +20, fg = -20)
    logits_correct = torch.full((B, num_classes, H, W), -20.0)
    logits_correct[:, 0] = 20.0
    logits_correct.requires_grad_(True)
    
    dice_fn = MultiClassSoftDiceLoss(num_classes=num_classes)
    loss_dice_corr = dice_fn(logits_correct, labels_2d)
    assert loss_dice_corr.item() < 0.05, f"Expected near-zero loss for perfect background, got {loss_dice_corr.item()}"
    loss_dice_corr.backward()
    assert torch.isfinite(logits_correct.grad).all()

    # Worst-case wrong prediction on pure background (model predicts foreground everywhere)
    logits_wrong = torch.full((B, num_classes, H, W), -20.0)
    logits_wrong[:, 3] = 20.0  # predicts scar everywhere
    logits_wrong.requires_grad_(True)
    
    loss_dice_wrong = dice_fn(logits_wrong, labels_2d)
    assert torch.isfinite(loss_dice_wrong)
    # With 4 foreground classes, 1 class having FP everywhere gives loss ~0.25 (1.0 / 4)
    assert loss_dice_wrong.item() > 0.20, f"Expected penalty > 0.20 for false-positive class, got {loss_dice_wrong.item()}"
    assert loss_dice_wrong.item() > 50 * loss_dice_corr.item(), "Wrong prediction loss must be significantly higher than correct background"
    loss_dice_wrong.backward()
    assert torch.isfinite(logits_wrong.grad).all()

    # 3D pure background volume
    B, D, H, W = 2, 8, 32, 32
    labels_3d = torch.zeros((B, D, H, W), dtype=torch.long)
    logits_3d = torch.randn((B, num_classes, D, H, W), requires_grad=True)
    loss_3d = dice_fn(logits_3d, labels_3d)
    assert torch.isfinite(loss_3d)
    loss_3d.backward()
    assert torch.isfinite(logits_3d.grad).all()


def test_loss_pure_foreground_slices():
    """Stress test: pure foreground slices (every voxel belongs to class c)."""
    num_classes = 5
    B, H, W = 2, 64, 64
    
    for c in range(1, num_classes):
        labels = torch.full((B, H, W), c, dtype=torch.long)
        
        # Test MultiClassSoftDiceLoss
        logits_mc = torch.randn((B, num_classes, H, W), requires_grad=True)
        l_mc = MultiClassSoftDiceLoss(num_classes=num_classes)(logits_mc, labels)
        assert torch.isfinite(l_mc)
        l_mc.backward()
        assert torch.isfinite(logits_mc.grad).all()
        
        # Test OneVsRestCompoundLoss
        logits_ovr = torch.randn((B, num_classes - 1, H, W), requires_grad=True)
        l_ovr = OneVsRestCompoundLoss(num_classes=num_classes)(logits_ovr, labels)
        assert torch.isfinite(l_ovr)
        l_ovr.backward()
        assert torch.isfinite(logits_ovr.grad).all()

        # Test FocalTverskyLoss
        logits_ft = torch.randn((B, num_classes, H, W), requires_grad=True)
        l_ft = FocalTverskyLoss(num_classes=num_classes)(logits_ft, labels)
        assert torch.isfinite(l_ft)
        l_ft.backward()
        assert torch.isfinite(logits_ft.grad).all()


def test_loss_extreme_logit_values_and_saturation():
    """Stress test: extreme logit values (+-100.0, +-500.0) simulating severe saturation."""
    num_classes = 5
    B, H, W = 2, 48, 48
    labels = torch.randint(0, num_classes, (B, H, W), dtype=torch.long)
    
    extreme_values = [100.0, -100.0, 500.0, -500.0]
    
    for val in extreme_values:
        # MultiClassSoftDiceLoss
        logits_mc = torch.full((B, num_classes, H, W), val, requires_grad=True)
        l_mc = MultiClassSoftDiceLoss(num_classes=num_classes)(logits_mc, labels)
        assert torch.isfinite(l_mc), f"SoftDiceLoss failed on extreme logits val={val}"
        l_mc.backward()
        assert torch.isfinite(logits_mc.grad).all(), f"SoftDiceLoss non-finite grad on val={val}"

        # OneVsRestCompoundLoss
        logits_ovr = torch.full((B, num_classes - 1, H, W), val, requires_grad=True)
        l_ovr = OneVsRestCompoundLoss(num_classes=num_classes)(logits_ovr, labels)
        assert torch.isfinite(l_ovr), f"OneVsRestCompoundLoss failed on extreme logits val={val}"
        l_ovr.backward()
        assert torch.isfinite(logits_ovr.grad).all(), f"OneVsRestCompoundLoss non-finite grad on val={val}"

        # FocalTverskyLoss
        logits_ft = torch.full((B, num_classes, H, W), val, requires_grad=True)
        l_ft = FocalTverskyLoss(num_classes=num_classes)(logits_ft, labels)
        assert torch.isfinite(l_ft), f"FocalTverskyLoss failed on extreme logits val={val}"
        l_ft.backward()
        assert torch.isfinite(logits_ft.grad).all(), f"FocalTverskyLoss non-finite grad on val={val}"


def test_loss_factory_and_registry():
    """Verify build_loss and LOSS_REGISTRY with varied hyperparameters."""
    for loss_name in ["ce", "dice", "ce_dice", "one_vs_rest_compound", "focal_tversky"]:
        loss_fn = build_loss(loss_name, num_classes=5)
        logits = torch.randn((2, 5, 32, 32), requires_grad=True)
        labels = torch.randint(0, 5, (2, 32, 32), dtype=torch.long)
        
        # OVR head expects either 4 or 5 channels (handled automatically)
        l = loss_fn(logits, labels)
        assert torch.isfinite(l), f"build_loss('{loss_name}') produced non-finite loss"
        l.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all(), f"build_loss('{loss_name}') produced non-finite grad"


# =============================================================================
# CATEGORY 2: ANATOMICAL POSTPROCESSING ADVERSARIAL STRESS TESTS
# =============================================================================

def test_postprocess_apical_scar_through_plane_connection():
    """Apical scar on slice 0 connecting through-plane to myocardium on slice 1."""
    # 3D Stack: 5 slices of 64x64
    vol = np.zeros((5, 64, 64), dtype=np.int16)
    
    # Slice 0: Pure apical transmural scar (12x12 = 144 voxels), NO myocardium on slice 0
    vol[0, 26:38, 26:38] = 3
    
    # Slice 1: Myocardium ring surrounding apex
    vol[1, 20:44, 20:44] = 2
    vol[1, 28:36, 28:36] = 0  # LV pool
    
    # Slices 2..4: Mid-ventricular to basal myocardium
    for z in range(2, 5):
        vol[z, 18:46, 18:46] = 2
        vol[z, 26:38, 26:38] = 0

    cleaned = enforce_anatomical_constraints(
        vol,
        scar_class=3,
        myo_class=2,
        spacing=(10.0, 1.25, 1.25),
        tolerance_mm=2.5,
        min_scar_volume_mm3=15.0,
    )
    
    # Apical scar on slice 0 MUST be fully preserved due to 3D connectivity to slice 1 myocardium
    assert np.all(cleaned[0, 26:38, 26:38] == 3), "Slice 0 apical scar was deleted despite through-plane connection!"


def test_postprocess_floating_scars_zero_myocardium_volume():
    """Floating scar clusters with zero myocardium across the entire volume must be eliminated."""
    vol = np.zeros((4, 50, 50), dtype=np.int16)
    vol[0, 10:20, 10:20] = 3
    vol[2, 30:40, 30:40] = 3
    vol[3, 5:15, 35:45] = 3
    
    cleaned = enforce_anatomical_constraints(vol, scar_class=3, myo_class=2, spacing=(8.0, 1.0, 1.0))
    assert not np.any(cleaned == 3), "Floating scars with zero myocardium were not suppressed!"
    assert np.all(cleaned == 0), "Volume should be entirely 0"


def test_postprocess_subendocardial_vs_transmural_scars():
    """Verify both subendocardial (inner border) and transmural (full wall) scars are preserved."""
    vol = np.zeros((3, 60, 60), dtype=np.int16)
    
    # Define circular/ring myocardium on slice 1 (outer radius 20, inner radius 12, center at 30,30)
    y, x = np.ogrid[:60, :60]
    dist_from_center = np.sqrt((x - 30)**2 + (y - 30)**2)
    myo_mask = (dist_from_center >= 12) & (dist_from_center <= 20)
    vol[1][myo_mask] = 2
    
    # Region 1 (Transmural scar, sector angle 0 to 45 deg): replaces whole wall (dist 12..20)
    angle = np.arctan2(y - 30, x - 30)
    transmural_mask = myo_mask & (angle >= 0) & (angle <= np.pi / 4)
    vol[1][transmural_mask] = 3
    
    # Region 2 (Subendocardial scar, sector angle -pi/2 to -pi/4): replaces only inner half (dist 12..16)
    subendo_mask = (dist_from_center >= 12) & (dist_from_center <= 16) & (angle >= -np.pi / 2) & (angle <= -np.pi / 4)
    vol[1][subendo_mask] = 3

    # Slice 0 & 2 also have myocardium
    vol[0][myo_mask] = 2
    vol[2][myo_mask] = 2

    cleaned = enforce_anatomical_constraints(
        vol,
        scar_class=3,
        myo_class=2,
        spacing=(5.0, 1.0, 1.0),
        tolerance_mm=2.5,
    )
    
    # Both transmural and subendocardial scars must be retained
    assert np.all(cleaned[1][transmural_mask] == 3), "Transmural scar was incorrectly eroded or removed!"
    assert np.all(cleaned[1][subendo_mask] == 3), "Subendocardial scar was incorrectly removed!"


def test_postprocess_patchy_noncontiguous_scars_and_speckle_removal():
    """Verify non-contiguous patchy scars: preserve true lesions, clean distant noise and tiny speckles."""
    vol = np.zeros((1, 80, 80), dtype=np.int16)
    
    # Myocardium wall: box from 20..60, thickness 6
    vol[0, 20:60, 20:60] = 2
    vol[0, 26:54, 26:54] = 0  # Blood pool inside
    
    # Patch A: Valid scar patch in left wall (15 voxels > min_scar_voxels=5)
    vol[0, 30:35, 20:23] = 3
    
    # Patch B: Valid scar patch in right wall (15 voxels > min_scar_voxels=5)
    vol[0, 30:35, 57:60] = 3
    
    # Patch C: Tiny scar speckle inside myocardium (2 voxels < 5)
    vol[0, 45, 22:24] = 3
    
    # Patch D: Tiny scar speckle in blood pool far from wall (2 voxels < 5)
    vol[0, 40, 40:42] = 3
    
    # Patch E: Large false positive scar artifact in distant corner (30 voxels at 5..10, 5..10)
    vol[0, 5:10, 5:11] = 3

    cleaned = enforce_anatomical_constraints(
        vol,
        scar_class=3,
        myo_class=2,
        spacing=None,
        dilation_voxels=1,
        min_scar_voxels=5,
    )
    
    # Patch A & B preserved
    assert np.all(cleaned[0, 30:35, 20:23] == 3), "Patch A should be preserved"
    assert np.all(cleaned[0, 30:35, 57:60] == 3), "Patch B should be preserved"
    
    # Patch C (tiny speckle inside wall) reverted to myocardium (class 2)
    assert np.all(cleaned[0, 45, 22:24] == 2), "Tiny speckle inside wall should revert to myocardium"
    
    # Patch D (tiny speckle in blood pool) eliminated to 0
    assert np.all(cleaned[0, 40, 40:42] == 0), "Blood pool speckle should be eliminated"
    
    # Patch E (distant false positive) eliminated to 0
    assert np.all(cleaned[0, 5:10, 5:11] == 0), "Distant false positive artifact should be eliminated"


def test_postprocess_2d_and_3d_torch_tensor_support():
    """Verify PyTorch tensor input/output preservation across 2D and 3D."""
    # 2D Tensor
    t2d = torch.zeros((50, 50), dtype=torch.long)
    t2d[15:35, 15:35] = 2
    t2d[20:25, 20:25] = 3  # Scar inside myo
    t2d[2:5, 2:5] = 3      # Distant noise
    
    res_2d = enforce_anatomical_constraints(t2d, scar_class=3, myo_class=2)
    assert isinstance(res_2d, torch.Tensor), "Expected torch.Tensor output for tensor input"
    assert res_2d.dtype == t2d.dtype
    assert (res_2d[20:25, 20:25] == 3).all()
    assert (res_2d[2:5, 2:5] == 0).all()

    # 3D Tensor
    t3d = torch.zeros((3, 40, 40), dtype=torch.int32)
    t3d[1, 10:30, 10:30] = 2
    t3d[1, 15:20, 15:20] = 3
    res_3d = enforce_anatomical_constraints(t3d, scar_class=3, myo_class=2)
    assert isinstance(res_3d, torch.Tensor)
    assert res_3d.dtype == t3d.dtype
    assert (res_3d[1, 15:20, 15:20] == 3).all()


# =============================================================================
# CATEGORY 3: METRIC CALIBRATION & NATURE METHODS 2024 BENCHMARK TESTS
# =============================================================================

def test_metrics_symmetric_true_negative_calibration():
    """Verify symmetric True Negative calibration (pred = empty, target = empty)."""
    pred_empty = np.zeros((64, 64), dtype=np.uint8)
    target_empty = np.zeros((64, 64), dtype=np.uint8)
    
    # Symmetric Dice & IoU -> 1.0
    dice_sym = dice_score_symmetric(pred_empty, target_empty, num_classes=5)
    iou_sym = iou_score_symmetric(pred_empty, target_empty, num_classes=5)
    for c in range(1, 5):
        assert dice_sym[c] == 1.0, f"Symmetric Dice for empty class {c} must be 1.0"
        assert iou_sym[c] == 1.0, f"Symmetric IoU for empty class {c} must be 1.0"

    # HD95 Symmetric TN -> 0.0 mm
    hd_tn = hd95_binary(pred_empty, target_empty, spacing=(1.0, 1.0), empty_value=0.0)
    assert hd_tn == 0.0, "HD95 for True Negative must be 0.0 mm"

    # Standard conditional Dice & IoU -> NaN
    dice_cond = dice_score(pred_empty, target_empty, num_classes=5, empty_value=float("nan"))
    iou_cond = iou_score(pred_empty, target_empty, num_classes=5, empty_value=float("nan"))
    for c in range(1, 5):
        assert np.isnan(dice_cond[c]), f"Conditional Dice for empty class {c} must be NaN"
        assert np.isnan(iou_cond[c]), f"Conditional IoU for empty class {c} must be NaN"


def test_metrics_dynamic_fov_penalty_complete_miss():
    """Verify HD95 dynamic FOV diagonal penalty on complete misses."""
    # 2D shape (200, 200), spacing (1.5, 1.5) -> FOV = sqrt((200*1.5)^2 + (200*1.5)^2) = 300 * sqrt(2) ≈ 424.264 mm
    pred_empty = np.zeros((200, 200), dtype=np.uint8)
    target_obj = np.zeros((200, 200), dtype=np.uint8)
    target_obj[90:110, 90:110] = 1
    
    hd_miss_2d = hd95_binary(pred_empty, target_obj, spacing=(1.5, 1.5))
    expected_fov_2d = np.sqrt((200 * 1.5)**2 + (200 * 1.5)**2)
    assert np.isclose(hd_miss_2d, expected_fov_2d), f"Expected {expected_fov_2d}, got {hd_miss_2d}"

    # 3D shape (16, 128, 128), spacing (8.0, 1.2, 1.2)
    pred_empty_3d = np.zeros((16, 128, 128), dtype=np.uint8)
    target_obj_3d = np.zeros((16, 128, 128), dtype=np.uint8)
    target_obj_3d[8, 60:70, 60:70] = 1
    
    hd_miss_3d = hd95_binary(pred_empty_3d, target_obj_3d, spacing=(8.0, 1.2, 1.2))
    expected_fov_3d = np.sqrt((16 * 8.0)**2 + (128 * 1.2)**2 + (128 * 1.2)**2)
    assert np.isclose(hd_miss_3d, expected_fov_3d), f"Expected {expected_fov_3d}, got {hd_miss_3d}"


def test_metrics_single_voxel_hd95():
    """Verify single-voxel HD95 calculation without empty-surface crash."""
    pred = np.zeros((50, 50), dtype=np.uint8)
    target = np.zeros((50, 50), dtype=np.uint8)
    
    pred[20, 20] = 1
    target[20, 25] = 1  # 5 voxels apart along axis 1
    
    hd = hd95_binary(pred, target, spacing=(2.0, 1.5))
    expected_dist = 5.0 * 1.5  # 7.5 mm
    assert np.isclose(hd, expected_dist), f"Expected distance {expected_dist} mm, got {hd}"


def test_clinical_scar_metrics_2d_vs_3d():
    """Verify calculate_scar_metrics dimensional safety (2D returns NaN, 3D returns mL/g)."""
    # 2D Slice: must NOT return physical 3D volume
    mask_2d = np.zeros((100, 100), dtype=np.int16)
    mask_2d[30:60, 30:60] = 2  # myo voxels
    mask_2d[40:50, 40:50] = 3  # 100 scar voxels inside
    
    m2d = calculate_scar_metrics(mask_2d, spacing=(1.0, 1.0))
    assert np.isnan(m2d.scar_volume_ml), "2D scar volume must be NaN"
    assert np.isnan(m2d.scar_mass_g), "2D scar mass must be NaN"
    assert m2d.scar_voxels == 100
    assert m2d.myocardium_voxels == 800  # (30*30 - 100) = 800
    assert m2d.scar_fraction_of_myo_plus_scar is not None

    # 3D Stack: 10 slices of 100x100, spacing (8.0, 1.25, 1.25) mm
    # 1 voxel volume = 8.0 * 1.25 * 1.25 = 12.5 mm³ = 0.0125 mL
    mask_3d = np.zeros((10, 100, 100), dtype=np.int16)
    mask_3d[3:7, 40:50, 40:50] = 3  # 4 * 100 = 400 scar voxels
    mask_3d[3:7, 30:40, 30:40] = 2  # 4 * 100 = 400 myo voxels
    
    m3d = calculate_scar_metrics(mask_3d, spacing=(8.0, 1.25, 1.25), tissue_density_g_per_ml=1.05)
    expected_vol = 400 * 12.5 / 1000.0  # 5.0 mL
    expected_mass = 5.0 * 1.05          # 5.25 g
    
    assert np.isclose(m3d.scar_volume_ml, expected_vol), f"Expected {expected_vol} mL, got {m3d.scar_volume_ml}"
    assert np.isclose(m3d.scar_mass_g, expected_mass), f"Expected {expected_mass} g, got {m3d.scar_mass_g}"
    assert np.isclose(m3d.scar_fraction_of_myo_plus_scar, 0.5)


# =============================================================================
# CATEGORY 4: ADDITIONAL ADVERSARIAL TOPOLOGY & NUMERICAL STRESS TESTS
# =============================================================================

def test_loss_odd_and_anisotropic_dimensions_and_huge_weights():
    """Verify loss functions on odd spatial shapes, single voxel batches, and large weights."""
    shapes = [
        (1, 5, 1, 1),
        (2, 5, 7, 13, 19),
        (1, 5, 117, 213),
    ]
    for shape in shapes:
        labels = torch.randint(0, 5, (shape[0],) + shape[2:], dtype=torch.long)
        logits = torch.randn(shape, requires_grad=True)
        
        # SoftDice
        l_sd = MultiClassSoftDiceLoss(num_classes=5)(logits, labels)
        assert torch.isfinite(l_sd)
        l_sd.backward()
        assert torch.isfinite(logits.grad).all()

        # OVR with huge pos_weight
        logits_ovr = torch.randn((shape[0], 4) + shape[2:], requires_grad=True)
        ovr = OneVsRestCompoundLoss(
            num_classes=5,
            pos_weight=[500.0, 1000.0, 2000.0, 500.0],
            class_weights=[0.1, 10.0, 50.0, 1.0],
        )
        l_ovr = ovr(logits_ovr, labels)
        assert torch.isfinite(l_ovr)
        l_ovr.backward()
        assert torch.isfinite(logits_ovr.grad).all()


def test_postprocess_distant_myo_slice_suppression():
    """Scar on slice 0 with myocardium only on distant slice 4 (distance = 40 mm > 2.5 mm) must be deleted."""
    vol = np.zeros((6, 50, 50), dtype=np.int16)
    vol[0, 20:30, 20:30] = 3  # Scar on slice 0
    vol[4, 20:30, 20:30] = 2  # Myo only on slice 4 (4 slices away)
    
    cleaned = enforce_anatomical_constraints(
        vol,
        scar_class=3,
        myo_class=2,
        spacing=(10.0, 1.0, 1.0),
        tolerance_mm=2.5,
    )
    # The scar on slice 0 is too far from slice 4 myocardium (40mm >> 2.5mm tolerance)
    assert not np.any(cleaned[0] == 3), "Distant scar on slice 0 must be deleted when myo is 40mm away!"


def test_postprocess_apical_scar_offset_xy_suppression():
    """Scar on slice 0 with myocardium on slice 1 but offset by 30 voxels (in-plane distance >> tolerance) must be deleted."""
    vol = np.zeros((3, 60, 60), dtype=np.int16)
    vol[0, 5:15, 5:15] = 3    # Scar at top-left corner
    vol[1, 40:55, 40:55] = 2  # Myocardium at bottom-right corner
    
    cleaned = enforce_anatomical_constraints(
        vol,
        scar_class=3,
        myo_class=2,
        spacing=(10.0, 1.0, 1.0),
        tolerance_mm=2.5,
    )
    assert not np.any(cleaned[0] == 3), "Scar at top-left with myo at bottom-right must be deleted!"


def test_postprocess_no_scar_or_empty_mask():
    """Empty mask or mask without scar must return immediately without mutation."""
    empty = np.zeros((10, 50, 50), dtype=np.int16)
    res_empty = enforce_anatomical_constraints(empty)
    assert np.all(res_empty == 0)

    no_scar = np.zeros((10, 50, 50), dtype=np.int16)
    no_scar[3, 20:30, 20:30] = 2  # Myocardium
    no_scar[3, 10:20, 10:20] = 1  # LV
    res_no_scar = enforce_anatomical_constraints(no_scar)
    assert np.array_equal(res_no_scar, no_scar)


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING ADVERSARIAL STRESS TESTS FOR MILESTONE 4 (R4)")
    print("=" * 70)
    
    print("\n[Category 1] Loss Functions & Gradient Backprop Under Stress...")
    test_loss_extreme_imbalance_1_vs_100k()
    print("  -> 1 vs 100k Extreme Imbalance (SoftDice, OVR, FocalTversky, CE+Dice): PASSED")
    test_loss_pure_background_slices()
    print("  -> Pure Background Slices & Negative Logit Bounds: PASSED")
    test_loss_pure_foreground_slices()
    print("  -> Pure Foreground Slices Across Classes: PASSED")
    test_loss_extreme_logit_values_and_saturation()
    print("  -> Extreme Logits (+-100, +-500) & Saturation Invariance: PASSED")
    test_loss_factory_and_registry()
    print("  -> Loss Factory & Registry Multi-Config Stability: PASSED")
    test_loss_odd_and_anisotropic_dimensions_and_huge_weights()
    print("  -> Odd/Anisotropic Shapes & Large Weight Scaling: PASSED")
    
    print("\n[Category 2] Anatomical Postprocessing Stress & Topological Edge Cases...")
    test_postprocess_apical_scar_through_plane_connection()
    print("  -> 3D Apical Transmural Scar Through-Plane Connection: PASSED")
    test_postprocess_floating_scars_zero_myocardium_volume()
    print("  -> Floating Scars with Zero Myocardium Elimination: PASSED")
    test_postprocess_subendocardial_vs_transmural_scars()
    print("  -> Subendocardial vs Transmural Full-Wall Preservation: PASSED")
    test_postprocess_patchy_noncontiguous_scars_and_speckle_removal()
    print("  -> Patchy Non-Contiguous Scars & Wall Speckle Reversion: PASSED")
    test_postprocess_2d_and_3d_torch_tensor_support()
    print("  -> 2D & 3D PyTorch Tensor & NumPy Array Interoperability: PASSED")
    test_postprocess_distant_myo_slice_suppression()
    print("  -> Distant Slice (>2.5mm through-plane) Scar Suppression: PASSED")
    test_postprocess_apical_scar_offset_xy_suppression()
    print("  -> In-Plane XY Offset (>2.5mm) Scar Suppression: PASSED")
    test_postprocess_no_scar_or_empty_mask()
    print("  -> Empty & No-Scar Mask Pass-Through: PASSED")
    
    print("\n[Category 3] Metric Calibration & Nature Methods 2024 Alignment...")
    test_metrics_symmetric_true_negative_calibration()
    print("  -> Symmetric True Negative Calibration (Dice=1.0, IoU=1.0, HD95=0.0): PASSED")
    test_metrics_dynamic_fov_penalty_complete_miss()
    print("  -> Dynamic Patient FOV HD95 Penalty on Complete Miss: PASSED")
    test_metrics_single_voxel_hd95()
    print("  -> Single-Voxel HD95 Robustness: PASSED")
    test_clinical_scar_metrics_2d_vs_3d()
    print("  -> Clinical Scar Volume/Mass 2D (NaN) vs 3D (mL/g) Guard: PASSED")
    
    print("\n" + "=" * 70)
    print("ALL ADVERSARIAL STRESS TESTS COMPLETED WITH 100% SUCCESS!")
    print("=" * 70)
