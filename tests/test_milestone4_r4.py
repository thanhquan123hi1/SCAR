"""Dedicated test suite for Milestone 4 (R4) - SCAR Cardiac MRI Refactoring.

Tests:
1. Numerical stability and finite gradients on empty slices for all loss functions.
2. Metrics Reloaded (2024) alignment: dynamic FOV diagonal scaling, symmetric TN handling, Median + IQR summary.
3. 3D structuring element connectivity, transmural apical scar preservation, and cardiac wall validation.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.loss import (
    MultiClassSoftDiceLoss,
    OneVsRestCompoundLoss,
    FocalTverskyLoss,
    ce_dice_loss,
    build_loss,
    masks_to_one_vs_rest,
)
from training.metrics import (
    compute_fov_diagonal,
    hd95_binary,
    dice_score,
    dice_score_symmetric,
    iou_score,
    iou_score_symmetric,
    calculate_scar_metrics,
)
from training.postprocess.anatomical import enforce_anatomical_constraints, postprocess_predictions


# =========================================================================
# 1. LOSS FUNCTION TESTS
# =========================================================================

def test_multiclass_soft_dice_empty_slice_stability():
    """SoftDiceLoss on empty slice (y=0) must evaluate to < 0.05 with zero NaNs."""
    loss_fn = MultiClassSoftDiceLoss(num_classes=5, smooth=1.0)
    
    # 2D empty slice: background logit high, foreground logits low
    logits_2d = torch.full((2, 5, 64, 64), -10.0)
    logits_2d[:, 0] = 10.0
    logits_2d = logits_2d.requires_grad_(True)
    labels_2d = torch.zeros((2, 64, 64), dtype=torch.long)
    
    loss_2d = loss_fn(logits_2d, labels_2d)
    assert not torch.isnan(loss_2d), "SoftDiceLoss returned NaN on 2D empty slice"
    assert loss_2d.item() < 0.05, f"Expected loss < 0.05 on empty slice, got {loss_2d.item()}"
    
    loss_2d.backward()
    assert logits_2d.grad is not None, "Gradient missing on 2D backward"
    assert torch.isfinite(logits_2d.grad).all(), "Non-finite gradients on 2D backward"

    # 3D empty volume
    logits_3d = torch.full((1, 5, 8, 32, 32), -10.0)
    logits_3d[:, 0] = 10.0
    logits_3d = logits_3d.requires_grad_(True)
    labels_3d = torch.zeros((1, 8, 32, 32), dtype=torch.long)
    
    loss_3d = loss_fn(logits_3d, labels_3d)
    assert not torch.isnan(loss_3d), "SoftDiceLoss returned NaN on 3D empty volume"
    assert loss_3d.item() < 0.05, f"Expected loss < 0.05 on 3D empty volume, got {loss_3d.item()}"
    
    loss_3d.backward()
    assert logits_3d.grad is not None
    assert torch.isfinite(logits_3d.grad).all()


def test_one_vs_rest_compound_loss_stability():
    """OneVsRestCompoundLoss on empty slice must evaluate to < 0.05 with finite gradients."""
    loss_fn = OneVsRestCompoundLoss(
        num_classes=5,
        bce_weight=0.5,
        dice_weight=0.5,
        focal_weight=1.0,
        focal_gamma=2.0,
        pos_weight=[1.5, 2.0, 3.0, 1.5],
        class_weights=[1.0, 1.2, 2.0, 1.0],
        smooth=1.0,
    )
    
    # 4 foreground channels with negative logits (model correctly predicts no foreground)
    logits = torch.full((2, 4, 64, 64), -10.0, requires_grad=True)
    labels = torch.zeros((2, 64, 64), dtype=torch.long)
    
    loss = loss_fn(logits, labels)
    assert not torch.isnan(loss), "OneVsRestCompoundLoss returned NaN on empty slice"
    assert loss.item() < 0.05, f"Expected loss < 0.05 on empty slice, got {loss.item()}"
    
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_tversky_loss_stability():
    """FocalTverskyLoss on empty slice must evaluate to < 0.05 with finite gradients."""
    loss_fn = FocalTverskyLoss(num_classes=5, alpha=0.3, beta=0.7, gamma=1.33, smooth=1.0)
    
    logits = torch.full((2, 5, 64, 64), -10.0)
    logits[:, 0] = 10.0
    logits = logits.requires_grad_(True)
    labels = torch.zeros((2, 64, 64), dtype=torch.long)
    
    loss = loss_fn(logits, labels)
    assert not torch.isnan(loss)
    assert loss.item() < 0.05, f"Expected FocalTversky < 0.05 on empty slice, got {loss.item()}"
    
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# =========================================================================
# 2. BENCHMARK METRIC & CALIBRATION TESTS (Nature Methods 2024)
# =========================================================================

def test_compute_fov_diagonal():
    """Verify physical FOV diagonal computation in 2D and 3D."""
    # 2D 256x256 at 1.0mm x 1.0mm: sqrt(256^2 + 256^2) = 256 * sqrt(2) ≈ 362.0387 mm
    diag_2d = compute_fov_diagonal((256, 256), spacing=(1.0, 1.0))
    assert np.isclose(diag_2d, 256.0 * np.sqrt(2.0))

    # 3D 16x192x192 at (10.0, 1.0, 1.0) mm: sqrt((16*10)^2 + 192^2 + 192^2) = sqrt(25600 + 36864 + 36864) = sqrt(99328) ≈ 315.1635 mm
    diag_3d = compute_fov_diagonal((16, 192, 192), spacing=(10.0, 1.0, 1.0))
    expected_3d = np.sqrt(160.0**2 + 192.0**2 + 192.0**2)
    assert np.isclose(diag_3d, expected_3d)


def test_hd95_dynamic_fov_and_symmetric_tn():
    """Verify HD95 dynamic FOV scaling and True Negative handling."""
    pred_empty = np.zeros((100, 100), dtype=np.int16)
    target_empty = np.zeros((100, 100), dtype=np.int16)
    
    # 1. Symmetric True Negative (both empty) -> 0.0 mm
    hd_tn = hd95_binary(pred_empty, target_empty, spacing=(1.0, 1.0), empty_value=0.0)
    assert hd_tn == 0.0, f"Expected 0.0 mm for True Negative, got {hd_tn}"

    # 2. Complete miss (target present, pred empty) -> dynamic FOV diagonal: sqrt(100^2 + 100^2) ≈ 141.4214 mm
    target_obj = np.zeros((100, 100), dtype=np.int16)
    target_obj[40:50, 40:50] = 1
    hd_miss = hd95_binary(pred_empty, target_obj, spacing=(1.0, 1.0))
    expected_fov = np.sqrt(100.0**2 + 100.0**2)
    assert np.isclose(hd_miss, expected_fov), f"Expected FOV diagonal {expected_fov}, got {hd_miss}"

    # 3. Explicit penalty override (e.g. 300.0 mm)
    hd_override = hd95_binary(pred_empty, target_obj, spacing=(1.0, 1.0), penalty_distance=300.0)
    assert hd_override == 300.0


def test_dice_and_iou_symmetric_tn():
    """Verify Dice and IoU symmetric True Negative handling."""
    pred_empty = np.zeros((50, 50), dtype=np.int16)
    target_empty = np.zeros((50, 50), dtype=np.int16)
    
    # Standard dice with NaN for empty
    d_nan = dice_score(pred_empty, target_empty, num_classes=4, empty_value=float("nan"))
    assert np.isnan(d_nan[1]) and np.isnan(d_nan[2]) and np.isnan(d_nan[3])

    # Symmetric dice with 1.0 for TN
    d_sym = dice_score_symmetric(pred_empty, target_empty, num_classes=4)
    assert d_sym[1] == 1.0 and d_sym[2] == 1.0 and d_sym[3] == 1.0

    # Symmetric IoU with 1.0 for TN
    iou_sym = iou_score_symmetric(pred_empty, target_empty, num_classes=4)
    assert iou_sym[1] == 1.0 and iou_sym[2] == 1.0 and iou_sym[3] == 1.0


# =========================================================================
# 3. ANATOMICAL POSTPROCESSING & 3D TRANSMURAL SCAR PRESERVATION
# =========================================================================

def test_3d_apical_scar_preservation_across_slices():
    """Apical scar on slice 0 must connect to myocardium on slice 1 via 3D connectivity and be PRESERVED."""
    # 3D volume: (D=4, H=60, W=60)
    vol = np.zeros((4, 60, 60), dtype=np.int16)
    
    # Slice 0 (Apex): Transmural scar patch (10x10 = 100 voxels), NO myocardium on this slice!
    vol[0, 25:35, 25:35] = 3  # Scar
    
    # Slice 1: Myocardium ring directly adjacent in through-plane (z=1)
    vol[1, 20:40, 20:40] = 2  # Myocardium
    
    # Slice 2 & 3: Normal myocardium
    vol[2, 20:40, 20:40] = 2
    vol[3, 20:40, 20:40] = 2

    # Run anatomical postprocessing with full 3D structuring element
    cleaned = enforce_anatomical_constraints(
        vol,
        scar_class=3,
        myo_class=2,
        spacing=(10.0, 1.0, 1.0),
        tolerance_mm=2.5,
        min_scar_volume_mm3=15.0,
    )
    
    # Slice 0 apical scar must be 100% PRESERVED
    assert np.all(cleaned[0, 25:35, 25:35] == 3), "Apical scar on slice 0 was incorrectly deleted!"
    print("  -> 3D Apical Transmural Scar Preservation: PASSED")


def test_floating_scar_suppression_without_myocardium():
    """Scar predictions with zero myocardium anywhere must be suppressed."""
    vol_no_myo = np.zeros((4, 60, 60), dtype=np.int16)
    vol_no_myo[0, 25:35, 25:35] = 3  # Floating scar without any myocardium in volume
    
    cleaned = enforce_anatomical_constraints(
        vol_no_myo,
        scar_class=3,
        myo_class=2,
        spacing=(10.0, 1.0, 1.0),
    )
    assert not np.any(cleaned == 3), "Floating scar without myocardium must be suppressed!"


def test_isolated_far_noise_suppression():
    """Scar floating far away from the cardiac wall must be removed."""
    vol = np.zeros((4, 60, 60), dtype=np.int16)
    vol[1, 20:40, 20:40] = 2  # Myocardium
    vol[1, 25:30, 25:30] = 3  # Scar inside myocardium
    vol[1, 2:5, 2:5] = 3      # Distant noise at corner (x=2..5, y=2..5)
    
    cleaned = enforce_anatomical_constraints(
        vol,
        scar_class=3,
        myo_class=2,
        spacing=(10.0, 1.0, 1.0),
        tolerance_mm=2.5,
    )
    assert np.all(cleaned[1, 25:30, 25:30] == 3), "Valid scar in wall must be preserved"
    assert not np.any(cleaned[1, 2:5, 2:5] == 3), "Distant noise must be suppressed"


if __name__ == "__main__":
    print("Running Milestone 4 (R4) dedicated tests...")
    test_multiclass_soft_dice_empty_slice_stability()
    test_one_vs_rest_compound_loss_stability()
    test_focal_tversky_loss_stability()
    test_compute_fov_diagonal()
    test_hd95_dynamic_fov_and_symmetric_tn()
    test_dice_and_iou_symmetric_tn()
    test_3d_apical_scar_preservation_across_slices()
    test_floating_scar_suppression_without_myocardium()
    test_isolated_far_noise_suppression()
    print("ALL MILESTONE 4 TESTS PASSED SUCCESSFULLY!")
