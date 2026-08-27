"""Adversarial stress test suite for Milestone 4 (R4) - Medical Benchmark Metrics & Evaluation Pipeline.

Author: Challenger 2 (Empirical Challenger)
Target Modules:
- training/metrics/__init__.py
- training/evaluate.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.metrics import (
    ScarMetrics,
    calculate_scar_metrics,
    compute_fov_diagonal,
    dice_score,
    dice_score_symmetric,
    hd95_binary,
    iou_score,
    iou_score_symmetric,
    mean_dice,
)
from training.evaluate import evaluate_split


# =============================================================================
# 1. SYMMETRIC TRUE NEGATIVE (TN) HANDLING
# =============================================================================

def test_symmetric_tn_hd95():
    """Verify HD95 returns 0.0 mm for True Negatives (both empty) across 2D, 3D, and various spacings."""
    spacings_to_test = [
        (1.0, 1.0),
        (0.5, 0.5),
        (1.0, 1.0, 10.0),
        (0.5, 0.5, 0.5),
        (2.0, 2.0, 2.0),
        (1.25, 1.25, 8.0),
    ]
    shapes_to_test = [
        (64, 64),
        (256, 256),
        (16, 192, 192),
        (8, 64, 64),
        (32, 128, 128),
    ]

    for spacing in spacings_to_test:
        dim = len(spacing)
        matching_shapes = [s for s in shapes_to_test if len(s) == dim]
        for shape in matching_shapes:
            pred_empty = np.zeros(shape, dtype=bool)
            target_empty = np.zeros(shape, dtype=bool)

            hd = hd95_binary(pred_empty, target_empty, spacing=spacing, empty_value=0.0)
            assert hd == 0.0, f"Failed for shape {shape}, spacing {spacing}: expected 0.0 mm, got {hd}"

            # Verify with custom empty_value (e.g. NaN)
            hd_custom = hd95_binary(pred_empty, target_empty, spacing=spacing, empty_value=float("nan"))
            assert np.isnan(hd_custom), f"Expected NaN when empty_value=nan, got {hd_custom}"
    print("  [1.1] Symmetric True Negative HD95: PASSED")


def test_symmetric_tn_dice_and_iou():
    """Verify Dice and IoU symmetric functions return 1.0 for all-empty classes."""
    pred_empty = np.zeros((16, 128, 128), dtype=np.int16)
    target_empty = np.zeros((16, 128, 128), dtype=np.int16)
    num_classes = 5

    # Symmetric dice -> 1.0 for all foreground classes 1..4
    dice_sym = dice_score_symmetric(pred_empty, target_empty, num_classes=num_classes)
    for c in range(1, num_classes):
        assert dice_sym[c] == 1.0, f"Class {c} expected Dice=1.0 for TN, got {dice_sym[c]}"

    # Symmetric IoU -> 1.0 for all foreground classes 1..4
    iou_sym = iou_score_symmetric(pred_empty, target_empty, num_classes=num_classes)
    for c in range(1, num_classes):
        assert iou_sym[c] == 1.0, f"Class {c} expected IoU=1.0 for TN, got {iou_sym[c]}"

    # Asymmetric / standard dice -> NaN
    dice_std = dice_score(pred_empty, target_empty, num_classes=num_classes, empty_value=float("nan"))
    for c in range(1, num_classes):
        assert np.isnan(dice_std[c]), f"Class {c} expected NaN for standard dice on TN, got {dice_std[c]}"
    print("  [1.2] Symmetric True Negative Dice & IoU: PASSED")


def test_partial_class_presence_tn_handling():
    """Verify that if class 1 is present (TP) and class 3 is absent (TN), class 3 gets TN value and class 1 gets exact overlap."""
    pred = np.zeros((64, 64), dtype=np.int16)
    target = np.zeros((64, 64), dtype=np.int16)

    # Class 1: 50% overlap (pred 20 voxels, target 20 voxels, 10 overlap -> Dice = 2*10/(20+20) = 0.5)
    pred[10:30, 10] = 1   # 20 voxels
    target[20:40, 10] = 1 # 20 voxels
    # Class 2: FN (target has 10 voxels, pred has 0 -> Dice = 0.0)
    target[50:60, 50] = 2
    # Class 3: TN (both empty -> Dice = 1.0)
    # Class 4: FP (pred has 10 voxels, target has 0 -> Dice = 0.0)
    pred[50:60, 50] = 4

    scores = dice_score_symmetric(pred, target, num_classes=5)
    assert np.isclose(scores[1], 0.5), f"Expected Dice(c=1)=0.5, got {scores[1]}"
    assert scores[2] == 0.0, f"Expected Dice(c=2)=0.0 (FN), got {scores[2]}"
    assert scores[3] == 1.0, f"Expected Dice(c=3)=1.0 (TN), got {scores[3]}"
    assert scores[4] == 0.0, f"Expected Dice(c=4)=0.0 (FP), got {scores[4]}"
    print("  [1.3] Partial Class Presence TN Handling: PASSED")


# =============================================================================
# 2. FALSE POSITIVE / FALSE NEGATIVE EDGE CASES
# =============================================================================

def test_single_voxel_fp_and_fn_hd95():
    """Verify single-voxel FP and single-voxel FN fall back to dynamic FOV diagonal."""
    shape = (16, 192, 192)
    spacing = (10.0, 1.0, 1.0)
    expected_fov = compute_fov_diagonal(shape, spacing)

    # 1. Single voxel FP (target empty, pred has 1 voxel)
    pred_fp = np.zeros(shape, dtype=bool)
    pred_fp[0, 0, 0] = True
    target_empty = np.zeros(shape, dtype=bool)

    hd_fp = hd95_binary(pred_fp, target_empty, spacing=spacing)
    assert np.isclose(hd_fp, expected_fov), f"FP expected FOV diagonal {expected_fov}, got {hd_fp}"

    # 2. Single voxel FN (pred empty, target has 1 voxel at center)
    pred_empty = np.zeros(shape, dtype=bool)
    target_fn = np.zeros(shape, dtype=bool)
    target_fn[8, 96, 96] = True

    hd_fn = hd95_binary(pred_empty, target_fn, spacing=spacing)
    assert np.isclose(hd_fn, expected_fov), f"FN expected FOV diagonal {expected_fov}, got {hd_fn}"
    print("  [2.1] Single Voxel FP and FN HD95: PASSED")


def test_single_voxel_tp_and_disjoint_points():
    """Verify single-voxel exact match and single-voxel distance computations."""
    shape = (20, 50, 50)
    spacing = (5.0, 1.0, 2.0)

    # 1. Single voxel exact match: Pred == Target
    p1 = np.zeros(shape, dtype=bool)
    t1 = np.zeros(shape, dtype=bool)
    p1[5, 25, 25] = True
    t1[5, 25, 25] = True
    hd_tp = hd95_binary(p1, t1, spacing=spacing)
    assert hd_tp == 0.0, f"Single voxel exact match must have HD95=0.0 mm, got {hd_tp}"

    # 2. Single voxel shift in x (dim 1): shift by 3 voxels = 3 * 1.0 = 3.0 mm
    p2 = np.zeros(shape, dtype=bool)
    t2 = np.zeros(shape, dtype=bool)
    p2[5, 25, 25] = True
    t2[5, 28, 25] = True
    hd_shift_x = hd95_binary(p2, t2, spacing=spacing)
    assert np.isclose(hd_shift_x, 3.0), f"Expected 3.0 mm, got {hd_shift_x}"

    # 3. Single voxel shift in z (dim 0): shift by 2 slices = 2 * 5.0 = 10.0 mm
    p3 = np.zeros(shape, dtype=bool)
    t3 = np.zeros(shape, dtype=bool)
    p3[5, 25, 25] = True
    t3[7, 25, 25] = True
    hd_shift_z = hd95_binary(p3, t3, spacing=spacing)
    assert np.isclose(hd_shift_z, 10.0), f"Expected 10.0 mm, got {hd_shift_z}"

    # 4. Opposite corners: (0,0,0) vs (D-1, H-1, W-1)
    p4 = np.zeros(shape, dtype=bool)
    t4 = np.zeros(shape, dtype=bool)
    p4[0, 0, 0] = True
    t4[shape[0]-1, shape[1]-1, shape[2]-1] = True
    hd_corners = hd95_binary(p4, t4, spacing=spacing)
    expected_corner_dist = np.sqrt(
        ((shape[0]-1) * spacing[0])**2 +
        ((shape[1]-1) * spacing[1])**2 +
        ((shape[2]-1) * spacing[2])**2
    )
    assert np.isclose(hd_corners, expected_corner_dist)
    # Must be strictly less than FOV diagonal
    fov_diag = compute_fov_diagonal(shape, spacing)
    assert hd_corners < fov_diag, f"Corner distance {hd_corners} should be < FOV diagonal {fov_diag}"
    print("  [2.2] Single Voxel TP, Shifts & Corner Bounds: PASSED")


def test_disjoint_distant_multiple_components():
    """Verify HD95 with multiple small disjoint clusters across anisotropic volume."""
    shape = (16, 128, 128)
    spacing = (8.0, 1.25, 1.25)

    pred = np.zeros(shape, dtype=bool)
    target = np.zeros(shape, dtype=bool)

    # 3 isolated 1-voxel or 2-voxel components in Pred
    pred[2, 20, 20] = True
    pred[8, 60, 60] = True
    pred[14, 100, 100] = True

    # 2 components in Target
    target[2, 22, 20] = True  # near pred comp 1 (2 * 1.25 = 2.5 mm)
    target[8, 60, 65] = True  # near pred comp 2 (5 * 1.25 = 6.25 mm)

    hd = hd95_binary(pred, target, spacing=spacing)
    assert hd is not None and not np.isnan(hd)
    assert hd > 0.0
    fov_diag = compute_fov_diagonal(shape, spacing)
    assert hd <= fov_diag, f"HD95 {hd} must not exceed FOV diagonal {fov_diag}"
    print("  [2.3] Disjoint Distant Multiple Components: PASSED")


# =============================================================================
# 3. DYNAMIC FOV DIAGONAL CALCULATION ACROSS ANISOTROPIC SPACINGS
# =============================================================================

def test_dynamic_fov_diagonal_formula():
    """Verify FOV diagonal matches exact mathematical formula across various shapes/spacings."""
    test_cases = [
        ((16, 192, 192), (10.0, 1.0, 1.0)),
        ((16, 192, 192), (1.0, 1.0, 10.0)),
        ((32, 256, 256), (0.5, 0.5, 0.5)),
        ((10, 100, 100), (2.0, 2.0, 2.0)),
        ((1, 256, 256), (5.0, 1.25, 1.25)),
        ((256, 256), (0.8, 0.8)),
        ((512, 512), (0.5, 0.5)),
        ((128, 128, 64), (1.5, 1.5, 3.0)),
        ((1, 1, 1), (1.0, 1.0, 1.0)),
    ]
    for shape, spacing in test_cases:
        diag = compute_fov_diagonal(shape, spacing)
        expected = np.sqrt(sum((dim * s)**2 for dim, s in zip(shape, spacing)))
        assert np.isclose(diag, expected), f"Expected {expected}, got {diag}"

        # Verify that for any random binary masks of this shape, HD95 never exceeds FOV diagonal
        rng = np.random.default_rng(42)
        p = rng.random(shape) > 0.98
        t = rng.random(shape) > 0.98

        hd = hd95_binary(p, t, spacing=spacing)
        if hd is not None:
            assert hd <= diag + 1e-6, f"HD95 {hd} exceeded FOV diagonal {diag} for shape {shape}, spacing {spacing}"
    print("  [3.1] Dynamic FOV Diagonal Mathematical Exactness: PASSED")


def test_hd95_never_exceeds_physical_fov_diagonal_under_random_stress():
    """Adversarial stress: generate 50 random extreme cases and verify HD95 <= FOV diagonal."""
    rng = np.random.default_rng(12345)
    for i in range(50):
        ndim = rng.choice([2, 3])
        if ndim == 2:
            shape = (int(rng.integers(10, 128)), int(rng.integers(10, 128)))
            spacing = (float(rng.uniform(0.3, 3.0)), float(rng.uniform(0.3, 3.0)))
        else:
            shape = (int(rng.integers(3, 20)), int(rng.integers(20, 80)), int(rng.integers(20, 80)))
            spacing = (float(rng.uniform(5.0, 15.0)), float(rng.uniform(0.5, 2.0)), float(rng.uniform(0.5, 2.0)))

        fov_diag = compute_fov_diagonal(shape, spacing)

        # Mode 1: Empty pred, non-empty target
        p1 = np.zeros(shape, dtype=bool)
        t1 = np.zeros(shape, dtype=bool)
        t1[rng.integers(0, shape[0]), rng.integers(0, shape[1])] = True
        hd1 = hd95_binary(p1, t1, spacing=spacing)
        assert np.isclose(hd1, fov_diag), f"Iter {i}: FN HD95 {hd1} != FOV diagonal {fov_diag}"

        # Mode 2: Non-empty pred, empty target
        p2 = np.zeros(shape, dtype=bool)
        t2 = np.zeros(shape, dtype=bool)
        p2[rng.integers(0, shape[0]), rng.integers(0, shape[1])] = True
        hd2 = hd95_binary(p2, t2, spacing=spacing)
        assert np.isclose(hd2, fov_diag), f"Iter {i}: FP HD95 {hd2} != FOV diagonal {fov_diag}"

        # Mode 3: Sparsely populated both
        p3 = rng.random(shape) > 0.99
        t3 = rng.random(shape) > 0.99
        hd3 = hd95_binary(p3, t3, spacing=spacing)
        if hd3 is not None:
            assert hd3 <= fov_diag + 1e-5, f"Iter {i}: Sparse HD95 {hd3} > FOV diagonal {fov_diag}"
    print("  [3.2] 50-Run Monte Carlo HD95 <= FOV Diagonal Constraint: PASSED")


# =============================================================================
# 4. DUAL-REPORTING & SUMMARY STATS IN EVALUATE_SPLIT
# =============================================================================

class MockModel(nn.Module):
    """Predicts a fixed class map or uses ground truth with controlled noise/shift."""
    def __init__(self, mode="healthy"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        B = x.shape[0]
        D, H, W = x.shape[2:]
        logits = torch.zeros((B, 5, D, H, W), device=x.device, dtype=torch.float32)
        logits[:, 0] = 5.0  # Background default

        if self.mode == "healthy":
            # Predict only Myocardium (class 2), no scar (class 3)
            logits[:, 2, :, 20:44, 20:44] = 10.0
        elif self.mode == "always_scar":
            # Predict Myocardium and Scar
            logits[:, 2, :, 20:44, 20:44] = 10.0
            logits[:, 3, :, 28:36, 28:36] = 15.0
        elif self.mode == "all_zero":
            # Predict all background
            logits[:, 0] = 20.0
        return logits


def test_dual_reporting_all_true_negatives_cohort():
    """Stress test: Cohort where 100% of subjects are True Negatives for scar (GT=0, Pred=0)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_root = tmp_path / "data"
        data_root.mkdir()
        out_dir = tmp_path / "out_all_tn"
        out_dir.mkdir()

        affine = np.diag([1.0, 1.0, 10.0, 1.0])
        rows = []

        # 5 healthy subjects
        for i in range(5):
            img = (np.random.rand(64, 64, 16) * 100).astype(np.float32)
            lbl = np.zeros((64, 64, 16), dtype=np.int16)
            lbl[20:44, 20:44, 4:12] = 2  # Myocardium ONLY, NO SCAR!

            img_name = f"sub_{i}_img.nii.gz"
            lbl_name = f"sub_{i}_lbl.nii.gz"
            nib.save(nib.Nifti1Image(img, affine), str(data_root / img_name))
            nib.save(nib.Nifti1Image(lbl, affine), str(data_root / lbl_name))

            rows.append({
                "subject_id": f"s_{i}",
                "record_id": f"rec_{i}",
                "image_path": img_name,
                "label_path": lbl_name,
                "has_label": True,
                "view": "SAX",
            })

        df_manifest = pd.DataFrame(rows)
        config = {
            "num_classes": 5,
            "view": "SAX",
            "preprocessing": {
                "target_shape": [64, 64, 16],
                "target_spacing": [1.0, 1.0, 10.0],
                "intensity_percentiles": [1.0, 99.0],
            },
            "postprocess": {"use_rules": False, "anatomical_constraint": False},
        }

        # Model predicts only myocardium (class 2), no scar (class 3) -> All 5 subjects are TN for scar
        model = MockModel(mode="healthy")
        subj_df, summ_df = evaluate_split(
            model=model,
            df=df_manifest,
            data_root=data_root,
            config=config,
            device=torch.device("cpu"),
            save_predictions=False,
            output_dir=out_dir,
        )

        assert len(subj_df) == 5
        # For scar: has_gt_scar should be False for all 5
        assert not subj_df["has_gt_scar"].any()
        # Overall Dice for scar must be 1.0 for all 5
        assert (subj_df["dice_scar"] == 1.0).all()
        # Overall HD95 for scar must be 0.0 mm for all 5
        assert (subj_df["hd95_scar_mm"] == 0.0).all()
        # Conditional Dice and HD95 must be all NaN
        assert subj_df["dice_conditional_scar"].isna().all()
        assert subj_df["hd95_conditional_scar_mm"].isna().all()

        # Check summary table
        summ_dict = summ_df.set_index("metric").to_dict(orient="index")

        # 1. Overall Dice scar: count=5, mean=1.0, std=0.0, median=1.0, iqr=0.0, q25=1.0, q75=1.0
        dice_scar_row = summ_dict["dice_scar"]
        assert dice_scar_row["count"] == 5
        assert dice_scar_row["mean"] == 1.0
        assert dice_scar_row["median"] == 1.0
        assert dice_scar_row["iqr"] == 0.0
        assert dice_scar_row["std"] == 0.0

        # 2. Overall HD95 scar: count=5, mean=0.0, std=0.0, median=0.0, iqr=0.0
        hd95_scar_row = summ_dict["hd95_scar_mm"]
        assert hd95_scar_row["count"] == 5
        assert hd95_scar_row["mean"] == 0.0
        assert hd95_scar_row["median"] == 0.0
        assert hd95_scar_row["iqr"] == 0.0

        # 3. Conditional Dice scar: count=0, mean=NaN, median=NaN, iqr=NaN
        cond_dice_scar = summ_dict["dice_conditional_scar"]
        assert cond_dice_scar["count"] == 0
        assert np.isnan(cond_dice_scar["mean"])
        assert np.isnan(cond_dice_scar["median"])
        assert np.isnan(cond_dice_scar["iqr"])
    print("  [4.1] Dual-Reporting 100% True Negative Cohort: PASSED")


def test_dual_reporting_mixed_cohort_accuracy():
    """Stress test: Mixed cohort with 2 TN, 2 TP (partial overlap), 1 FN, 1 FP.
    Verify exact statistical values: Median, IQR, Q25, Q75, Mean, Std, Count."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_root = tmp_path / "data"
        data_root.mkdir()
        out_dir = tmp_path / "out_mixed"
        out_dir.mkdir()

        affine = np.diag([1.0, 1.0, 10.0, 1.0])

        class PredefinedModel(nn.Module):
            def __init__(self, sequence):
                super().__init__()
                self.sequence = sequence
                self.call_count = 0

            def forward(self, x):
                mode = self.sequence[self.call_count]
                self.call_count += 1
                B = x.shape[0]
                D, H, W = x.shape[2:]
                logits = torch.zeros((B, 5, D, H, W), dtype=torch.float32)
                logits[:, 0] = 5.0
                if mode == "scar":
                    logits[:, 2, :, 20:44, 20:44] = 10.0
                    logits[:, 3, :, 28:36, 28:36] = 15.0
                elif mode == "healthy":
                    logits[:, 2, :, 20:44, 20:44] = 10.0
                return logits

        # sub 0: GT no scar
        img0 = np.full((64, 64, 16), 1.0, dtype=np.float32)
        lbl0 = np.zeros((64, 64, 16), dtype=np.int16)
        lbl0[20:44, 20:44, :] = 2
        nib.save(nib.Nifti1Image(img0, affine), str(data_root / "s0_img.nii.gz"))
        nib.save(nib.Nifti1Image(lbl0, affine), str(data_root / "s0_lbl.nii.gz"))

        # sub 1: GT no scar
        img1 = np.full((64, 64, 16), 2.0, dtype=np.float32)
        lbl1 = np.zeros((64, 64, 16), dtype=np.int16)
        lbl1[20:44, 20:44, :] = 2
        nib.save(nib.Nifti1Image(img1, affine), str(data_root / "s1_img.nii.gz"))
        nib.save(nib.Nifti1Image(lbl1, affine), str(data_root / "s1_lbl.nii.gz"))

        # sub 2: GT has scar
        img2 = np.full((64, 64, 16), 20.0, dtype=np.float32)
        lbl2 = np.zeros((64, 64, 16), dtype=np.int16)
        lbl2[20:44, 20:44, :] = 2
        lbl2[28:36, 28:36, :] = 3
        nib.save(nib.Nifti1Image(img2, affine), str(data_root / "s2_img.nii.gz"))
        nib.save(nib.Nifti1Image(lbl2, affine), str(data_root / "s2_lbl.nii.gz"))

        # sub 3: GT has scar
        img3 = np.full((64, 64, 16), 25.0, dtype=np.float32)
        lbl3 = np.zeros((64, 64, 16), dtype=np.int16)
        lbl3[20:44, 20:44, :] = 2
        lbl3[28:36, 28:36, :] = 3
        nib.save(nib.Nifti1Image(img3, affine), str(data_root / "s3_img.nii.gz"))
        nib.save(nib.Nifti1Image(lbl3, affine), str(data_root / "s3_lbl.nii.gz"))

        # sub 4: GT has scar
        img4 = np.full((64, 64, 16), 3.0, dtype=np.float32)
        lbl4 = np.zeros((64, 64, 16), dtype=np.int16)
        lbl4[20:44, 20:44, :] = 2
        lbl4[28:36, 28:36, :] = 3
        nib.save(nib.Nifti1Image(img4, affine), str(data_root / "s4_img.nii.gz"))
        nib.save(nib.Nifti1Image(lbl4, affine), str(data_root / "s4_lbl.nii.gz"))

        # sub 5: GT has NO scar
        img5 = np.full((64, 64, 16), 30.0, dtype=np.float32)
        lbl5 = np.zeros((64, 64, 16), dtype=np.int16)
        lbl5[20:44, 20:44, :] = 2
        nib.save(nib.Nifti1Image(img5, affine), str(data_root / "s5_img.nii.gz"))
        nib.save(nib.Nifti1Image(lbl5, affine), str(data_root / "s5_lbl.nii.gz"))

        manifest = pd.DataFrame([
            {"subject_id": f"s{i}", "record_id": f"rec_{i}", "image_path": f"s{i}_img.nii.gz", "label_path": f"s{i}_lbl.nii.gz", "has_label": True, "view": "SAX"}
            for i in range(6)
        ])

        config = {
            "num_classes": 5,
            "view": "SAX",
            "preprocessing": {
                "target_shape": [64, 64, 16],
                "target_spacing": [1.0, 1.0, 10.0],
                "intensity_percentiles": None,
            },
            "postprocess": {"use_rules": False, "anatomical_constraint": False},
        }

        model = PredefinedModel(["healthy", "healthy", "scar", "scar", "healthy", "scar"])
        subj_df, summ_df = evaluate_split(
            model=model,
            df=manifest,
            data_root=data_root,
            config=config,
            device=torch.device("cpu"),
            save_predictions=False,
            output_dir=out_dir,
        )

        assert len(subj_df) == 6
        # Expected Dice values for scar:
        # s0 (TN): 1.0
        # s1 (TN): 1.0
        # s2 (TP): 1.0
        # s3 (TP): 1.0
        # s4 (FN): 0.0
        # s5 (FP): 0.0
        overall_dice_scar = subj_df["dice_scar"].tolist()
        assert overall_dice_scar == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0], f"Got {overall_dice_scar}"

        cond_dice_scar = subj_df["dice_conditional_scar"].dropna().tolist()
        assert cond_dice_scar == [1.0, 1.0, 0.0], f"Got {cond_dice_scar}"

        summ_dict = summ_df.set_index("metric").to_dict(orient="index")

        # Check Overall Dice summary
        ov_row = summ_dict["dice_scar"]
        assert ov_row["count"] == 6
        assert np.isclose(ov_row["mean"], 4.0 / 6.0)
        assert np.isclose(ov_row["median"], 1.0)
        assert np.isclose(ov_row["q25"], 0.25)
        assert np.isclose(ov_row["q75"], 1.0)
        assert np.isclose(ov_row["iqr"], 0.75)

        # Check Conditional Dice summary
        cd_row = summ_dict["dice_conditional_scar"]
        assert cd_row["count"] == 3
        assert np.isclose(cd_row["mean"], 2.0 / 3.0)
        assert np.isclose(cd_row["median"], 1.0)
        assert np.isclose(cd_row["q25"], 0.5)
        assert np.isclose(cd_row["q75"], 1.0)
        assert np.isclose(cd_row["iqr"], 0.5)
    print("  [4.2] Dual-Reporting Mixed Cohort Statistical Exactness: PASSED")


def test_dual_reporting_single_subject_dataset():
    """Verify single-subject cohort does not crash and handles std=NaN correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_root = tmp_path / "data"
        data_root.mkdir()
        out_dir = tmp_path / "out_single"
        out_dir.mkdir()

        affine = np.diag([1.0, 1.0, 10.0, 1.0])
        img = np.full((64, 64, 16), 1.0, dtype=np.float32)
        lbl = np.zeros((64, 64, 16), dtype=np.int16)
        lbl[20:44, 20:44, :] = 2
        lbl[28:36, 28:36, :] = 3

        nib.save(nib.Nifti1Image(img, affine), str(data_root / "s0_img.nii.gz"))
        nib.save(nib.Nifti1Image(lbl, affine), str(data_root / "s0_lbl.nii.gz"))

        manifest = pd.DataFrame([
            {"subject_id": "s0", "record_id": "rec_0", "image_path": "s0_img.nii.gz", "label_path": "s0_lbl.nii.gz", "has_label": True, "view": "SAX"}
        ])

        config = {
            "num_classes": 5,
            "view": "SAX",
            "preprocessing": {
                "target_shape": [64, 64, 16],
                "target_spacing": [1.0, 1.0, 10.0],
                "intensity_percentiles": None,
            },
            "postprocess": {"use_rules": False, "anatomical_constraint": False},
        }

        subj_df, summ_df = evaluate_split(
            model=MockModel(mode="always_scar"),
            df=manifest,
            data_root=data_root,
            config=config,
            device=torch.device("cpu"),
            save_predictions=False,
            output_dir=out_dir,
        )

        assert len(subj_df) == 1
        assert len(summ_df) > 0
        summ_dict = summ_df.set_index("metric").to_dict(orient="index")
        assert summ_dict["dice_scar"]["count"] == 1
        assert np.isclose(summ_dict["dice_scar"]["mean"], 1.0)
        assert np.isnan(summ_dict["dice_scar"]["std"])  # N=1 sample std is NaN in pandas
    print("  [4.3] Dual-Reporting Single Subject Edge Case: PASSED")


# =============================================================================
# 5. CLINICAL SCAR QUANTIFICATION EDGE CASES
# =============================================================================

def test_clinical_scar_metrics_2d_vs_3d():
    """Verify calculate_scar_metrics returns NaN for 2D inputs and exact volume/mass for 3D SAX."""
    # 1. 2D single slice (ndim=2)
    mask_2d = np.zeros((100, 100), dtype=np.int16)
    mask_2d[30:70, 30:70] = 2  # Myo
    mask_2d[40:50, 40:50] = 3  # 100 voxels of scar
    metrics_2d = calculate_scar_metrics(mask_2d, spacing=(1.0, 1.0))
    assert np.isnan(metrics_2d.scar_volume_ml), "2D volume must be NaN"
    assert np.isnan(metrics_2d.scar_mass_g), "2D mass must be NaN"
    assert metrics_2d.scar_voxels == 100

    # 2. 3D stack (ndim=3)
    mask_3d = np.zeros((10, 100, 100), dtype=np.int16)
    mask_3d[0:10, 30:70, 30:70] = 2
    mask_3d[0:10, 40:50, 40:50] = 3  # 10 * 10 * 10 = 1000 voxels
    metrics_3d = calculate_scar_metrics(mask_3d, spacing=(10.0, 1.0, 1.0), tissue_density_g_per_ml=1.05)
    assert np.isclose(metrics_3d.scar_volume_ml, 10.0)
    assert np.isclose(metrics_3d.scar_mass_g, 10.5)
    assert metrics_3d.scar_voxels == 1000

    # 3. 3D stack with 0 scar
    mask_3d_zero = np.zeros((10, 100, 100), dtype=np.int16)
    mask_3d_zero[:, 20:30, 20:30] = 2
    metrics_zero = calculate_scar_metrics(mask_3d_zero, spacing=(10.0, 1.0, 1.0))
    assert metrics_zero.scar_volume_ml == 0.0
    assert metrics_zero.scar_mass_g == 0.0
    assert metrics_zero.scar_fraction_of_myo_plus_scar == 0.0
    print("  [5.1] Clinical Scar Metrics 2D vs 3D & Density Scaling: PASSED")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING ADVERSARIAL STRESS TEST SUITE - CHALLENGER 2 (MILESTONE 4)")
    print("=================================================================")
    test_symmetric_tn_hd95()
    test_symmetric_tn_dice_and_iou()
    test_partial_class_presence_tn_handling()
    test_single_voxel_fp_and_fn_hd95()
    test_single_voxel_tp_and_disjoint_points()
    test_disjoint_distant_multiple_components()
    test_dynamic_fov_diagonal_formula()
    test_hd95_never_exceeds_physical_fov_diagonal_under_random_stress()
    test_dual_reporting_all_true_negatives_cohort()
    test_dual_reporting_mixed_cohort_accuracy()
    test_dual_reporting_single_subject_dataset()
    test_clinical_scar_metrics_2d_vs_3d()
    print("=================================================================")
    print("ALL ADVERSARIAL STRESS TESTS PASSED WITH 100% EMPIRICAL RIGOR!")
    print("=================================================================")
