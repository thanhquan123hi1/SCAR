"""Challenger M1 Stress Test Harness.

Adversarial testing suite for:
- preprocessing/preprocessing.py (preprocess_mask, invert_spatial_mask, center_crop_or_pad, etc.)
- preprocessing/build_splits.py (verify_split_independence, DataLeakageError)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.preprocessing import (
    CenterTransform,
    SpatialTransform,
    center_crop_or_pad,
    invert_center_crop_or_pad,
    invert_spatial_mask,
    preprocess_mask,
    preprocess_spatial,
    shape_for_spacing,
)
from preprocessing.build_splits import DataLeakageError, verify_split_independence


def test_extreme_anisotropic_resampling():
    """Stress test 1: Highly anisotropic voxel spacings (10x downsampling and upsampling)."""
    print("[Attack 1] Testing extreme anisotropic voxel spacings (10x scale shifts)...")

    # 1A: 10x downsampling in X, 5x upsampling in Y
    # Shape (100, 100), source spacing (1.0, 10.0), target spacing (10.0, 2.0)
    # Expected resampled shape: (100 * 1.0 / 10.0, 100 * 10.0 / 2.0) = (10, 500)
    source_shape = (100, 100)
    source_spacing = (1.0, 10.0)
    target_spacing = (10.0, 2.0)
    target_shape = (10, 500)

    mask_2d = np.zeros(source_shape, dtype=np.int16)
    mask_2d[40:60, 40:60] = 2  # 20x20 block of class 2
    mask_2d[48:52, 48:52] = 3  # 4x4 block of class 3 inside

    res = preprocess_mask(
        mask_2d,
        source_spacing=source_spacing,
        target_spacing=target_spacing,
        target_shape=target_shape,
    )
    assert res.shape == target_shape, f"Shape mismatch: {res.shape} vs {target_shape}"
    assert res.dtype in (np.int64, np.int32, np.int16)
    assert set(np.unique(res)).issubset({0, 2, 3})
    assert (res == 2).sum() > 0, "Class 2 lost in anisotropic resampling"
    assert (res == 3).sum() > 0, "Class 3 lost in anisotropic resampling"

    # 1B: 3D Thick-slice Anisotropic volume (16, 200, 200) with (10.0, 1.25, 1.25) mm -> (1.5, 1.5, 1.5) mm
    # In-plane downsample (1.25 -> 1.5), Through-plane 6.67x upsample (10.0 -> 1.5)
    source_shape_3d = (16, 200, 200)
    source_sp_3d = (10.0, 1.25, 1.25)
    target_sp_3d = (1.5, 1.5, 1.5)
    expected_3d_shape = shape_for_spacing(source_shape_3d, source_sp_3d, target_sp_3d)
    # (16 * 10.0 / 1.5 = 107, 200 * 1.25 / 1.5 = 167, 200 * 1.25 / 1.5 = 167)
    assert expected_3d_shape == (107, 167, 167), f"Unexpected 3D shape {expected_3d_shape}"

    mask_3d = np.zeros(source_shape_3d, dtype=np.int16)
    mask_3d[5:10, 80:120, 80:120] = 1  # LV
    mask_3d[6:9, 90:110, 90:110] = 2  # Myo
    mask_3d[7, 95:100, 95:100] = 3  # Scar

    res_3d = preprocess_mask(
        mask_3d,
        source_spacing=source_sp_3d,
        target_spacing=target_sp_3d,
        target_shape=expected_3d_shape,
    )
    assert res_3d.shape == expected_3d_shape
    assert set(np.unique(res_3d)) == {0, 1, 2, 3}
    print("  -> Extreme anisotropic 2D & 3D resampling: PASSED")


def test_concentric_multiclass_mutual_exclusivity():
    """Stress test 2: 5 tightly nested concentric classes testing mutual exclusivity & no overlap."""
    print("[Attack 2] Testing 5 tightly nested concentric classes...")

    # Construct Russian-doll concentric square rings:
    # Class 0: Background
    # Class 1: Outer ring [20:80, 20:80]
    # Class 2: Ring 2 [30:70, 30:70]
    # Class 3: Ring 3 [40:60, 40:60]
    # Class 4: Ring 4 [45:55, 45:55]
    # Class 5: Innermost core [48:52, 48:52]
    grid = np.zeros((100, 100), dtype=np.int16)
    grid[20:80, 20:80] = 1
    grid[30:70, 30:70] = 2
    grid[40:60, 40:60] = 3
    grid[45:55, 45:55] = 4
    grid[48:52, 48:52] = 5

    # Verify input classes
    assert set(np.unique(grid)) == {0, 1, 2, 3, 4, 5}

    # Test under multiple scale ratios: 2x downsample, 2.5x downsample, 3x upsample
    for scale_factor in [0.5, 0.4, 1.8, 3.0]:
        target_sp = (1.0 / scale_factor, 1.0 / scale_factor)
        target_shape = shape_for_spacing(grid.shape, (1.0, 1.0), target_sp)
        res = preprocess_mask(
            grid,
            source_spacing=(1.0, 1.0),
            target_spacing=target_sp,
            target_shape=target_shape,
        )

        # 1. Output must be purely integer / discrete
        assert np.all(np.equal(np.mod(res, 1), 0))
        # 2. All 5 classes must be preserved
        classes_present = set(np.unique(res))
        assert {0, 1, 2, 3, 4, 5}.issubset(classes_present), (
            f"Scale {scale_factor} dropped classes: missing {{0,1,2,3,4,5}} - {classes_present}"
        )
        # 3. Centroid preservation for the innermost core (class 5)
        core_pts = np.argwhere(res == 5)
        cy, cx = core_pts.mean(axis=0)
        expected_center = (target_shape[0] - 1) / 2.0
        assert abs(cy - expected_center) < 1.0 and abs(cx - expected_center) < 1.0, (
            f"Centroid shifted for class 5 at scale {scale_factor}: got ({cy}, {cx}), expected ~{expected_center}"
        )

    print("  -> 5 tightly nested concentric classes across multiple scales: PASSED")


def test_micro_lesions_and_boundary_cases():
    """Stress test 3: Micro-lesions (single voxel, 2-voxel), corners, empty masks, non-contiguous labels."""
    print("[Attack 3] Testing micro-lesions, corner positions, empty masks, and non-contiguous labels...")

    # 3A: Empty mask (all zeros)
    empty = np.zeros((64, 64), dtype=np.int16)
    res_empty = preprocess_mask(
        empty,
        source_spacing=(1.0, 1.0),
        target_spacing=(2.0, 2.0),
        target_shape=(32, 32),
    )
    assert res_empty.shape == (32, 32)
    assert np.all(res_empty == 0)

    # 3B: Uniform non-zero mask (all class 4)
    uniform = np.full((64, 64), 4, dtype=np.int16)
    res_uniform = preprocess_mask(
        uniform,
        source_spacing=(1.0, 1.0),
        target_spacing=(0.5, 0.5),
        target_shape=(128, 128),
    )
    assert res_uniform.shape == (128, 128)
    assert np.all(res_uniform == 4)

    # 3C: Single-voxel lesion at (0, 0) and at center
    corner_mask = np.zeros((100, 100), dtype=np.int16)
    corner_mask[0, 0] = 3  # corner micro-lesion
    corner_mask[50, 50] = 2  # center micro-lesion
    res_corner = preprocess_mask(
        corner_mask,
        source_spacing=(1.0, 1.0),
        target_spacing=(2.0, 2.0),
        target_shape=(50, 50),
    )
    assert (res_corner == 3).sum() >= 1, "Corner 1-voxel micro lesion lost in 2x downsampling!"
    assert (res_corner == 2).sum() >= 1, "Center 1-voxel micro lesion lost in 2x downsampling!"
    assert res_corner[0, 0] == 3, f"Expected res_corner[0, 0] == 3, got {res_corner[0, 0]}"

    # 3D: Non-contiguous label IDs (e.g. {0, 7, 42, 999})
    non_contig = np.zeros((60, 60), dtype=np.int16)
    non_contig[10:25, 10:25] = 7
    non_contig[30:45, 30:45] = 42
    non_contig[50:55, 50:55] = 999

    res_non_contig = preprocess_mask(
        non_contig,
        source_spacing=(1.0, 1.0),
        target_spacing=(1.5, 1.5),
        target_shape=(40, 40),
    )
    assert set(np.unique(res_non_contig)) == {0, 7, 42, 999}, (
        f"Non-contiguous label preservation failed: {np.unique(res_non_contig)}"
    )
    print("  -> Micro-lesions, corners, empty/uniform, non-contiguous labels: PASSED")


def test_inverse_spatial_mask_roundtrip():
    """Stress test 4: Invert spatial mask fidelity and round-trip consistency."""
    print("[Attack 4] Testing inverse spatial mask round-trip and crop/pad recovery...")

    # Create original mask (180, 220)
    original_shape = (180, 220)
    orig_mask = np.zeros(original_shape, dtype=np.int16)
    orig_mask[60:120, 80:140] = 1  # LV
    orig_mask[70:110, 90:130] = 2  # Myo
    orig_mask[85:95, 105:115] = 3  # Scar (10x10 = 100 voxels)

    # 4A: Preprocess spatial image + mask
    dummy_img = np.random.uniform(0.1, 1.0, original_shape).astype(np.float32)
    processed_img, transform = preprocess_spatial(
        dummy_img,
        source_spacing=(1.2, 1.4),
        target_spacing=(1.0, 1.0),
        target_shape=(256, 256),
    )
    processed_mask = preprocess_mask(
        orig_mask,
        source_spacing=(1.2, 1.4),
        target_spacing=(1.0, 1.0),
        target_shape=(256, 256),
    )

    # 4B: Invert prediction mask back to original geometry
    restored_mask = invert_spatial_mask(processed_mask, transform)

    assert restored_mask.shape == original_shape, (
        f"Restored shape mismatch: {restored_mask.shape} vs {original_shape}"
    )
    assert set(np.unique(restored_mask)) == {0, 1, 2, 3}, (
        f"Class loss in roundtrip: {np.unique(restored_mask)}"
    )

    # Check scar centroid preservation in original grid
    orig_scar_pts = np.argwhere(orig_mask == 3)
    rest_scar_pts = np.argwhere(restored_mask == 3)
    orig_c = orig_scar_pts.mean(axis=0)
    rest_c = rest_scar_pts.mean(axis=0)
    centroid_dist = np.linalg.norm(orig_c - rest_c)
    assert centroid_dist < 1.0, f"Centroid shift too large: {centroid_dist:.2f} voxels"

    # 4C: Invert mask with tiny 1-voxel lesion
    tiny_mask = np.zeros((256, 256), dtype=np.int16)
    tiny_mask[128, 128] = 3
    restored_tiny = invert_spatial_mask(tiny_mask, transform)
    assert (restored_tiny == 3).sum() >= 1, "Tiny 1-voxel lesion lost during invert_spatial_mask!"
    print("  -> Inverse spatial mask round-trip and micro-lesion preservation: PASSED")


def test_split_independence_adversarial_cases():
    """Stress test 5: Adversarial and randomized patient split leakage scenarios."""
    print("[Attack 5] Testing verify_split_independence under adversarial configurations...")

    # 5A: Multi-view cross-leakage matrix
    # Patient 101 has SAX in train, but 2CH in validation
    # Patient 102 has 4CH in validation, but RAS in test
    # Patient 103 is clean (all 4 views in train)
    # Patient 104 is clean (all 4 views in validation)
    # Patient 105 is clean (all 4 views in test)
    records = []
    # Patient 101 (leaked train/val)
    records.append({"subject_id": "101", "view": "SAX", "split": "train"})
    records.append({"subject_id": "101", "view": "2CH", "split": "validation"})
    # Patient 102 (leaked val/test)
    records.append({"subject_id": "102", "view": "4CH", "split": "validation"})
    records.append({"subject_id": "102", "view": "RAS", "split": "test"})
    # Clean patients
    for v in ["SAX", "2CH", "4CH", "RAS"]:
        records.append({"subject_id": "103", "view": v, "split": "train"})
        records.append({"subject_id": "104", "view": v, "split": "validation"})
        records.append({"subject_id": "105", "view": v, "split": "test"})

    df_adversarial = pd.DataFrame(records)

    # In strict mode, must raise DataLeakageError
    leak_caught = False
    try:
        verify_split_independence(df_adversarial, strict=True)
    except DataLeakageError as e:
        leak_caught = True
        err_msg = str(e)
        assert "101" in err_msg and "102" in err_msg, f"Missing leaked patient IDs in error message: {err_msg}"
        assert "train" in err_msg and "validation" in err_msg and "test" in err_msg
    assert leak_caught, "Failed to catch multi-patient cross-view leakage!"

    # In non-strict mode, must return report with is_independent=False and 2 leakage records
    report = verify_split_independence(df_adversarial, strict=False)
    assert report["is_independent"] is False
    assert len(report["leakages"]) == 2

    # 5B: Large randomized synthetic cohort (500 patients, 4 views each)
    np.random.seed(42)
    clean_records = []
    patients = [f"PAT_{i:04d}" for i in range(500)]
    split_choices = ["train"] * 350 + ["validation"] * 75 + ["test"] * 75
    np.random.shuffle(split_choices)

    for pid, split in zip(patients, split_choices):
        # assign 1 to 4 views randomly
        n_views = np.random.randint(1, 5)
        views = np.random.choice(["SAX", "2CH", "4CH", "RAS"], size=n_views, replace=False)
        for v in views:
            clean_records.append({"subject_id": pid, "view": v, "split": split})

    df_large_clean = pd.DataFrame(clean_records)
    clean_report = verify_split_independence(df_large_clean, strict=True)
    assert clean_report["is_independent"] is True
    assert clean_report["patient_counts"] == {"train": 350, "validation": 75, "test": 75}

    # Inject 1 rogue row: PAT_0000 (train) gets a test row for RAS
    df_large_injected = pd.concat([
        df_large_clean,
        pd.DataFrame([{"subject_id": "PAT_0000", "view": "RAS", "split": "test"}]),
    ], ignore_index=True)

    injected_caught = False
    try:
        verify_split_independence(df_large_injected, strict=True)
    except DataLeakageError as e:
        injected_caught = True
        assert "PAT_0000" in str(e)
    assert injected_caught, "Rogue injected patient row was not caught in large cohort!"

    # 5C: Edge cases: Missing columns, empty DataFrame, integer subject IDs vs strings
    # Missing column
    df_no_split = pd.DataFrame([{"subject_id": "001"}])
    try:
        verify_split_independence(df_no_split)
        assert False, "Should raise ValueError for missing split column"
    except ValueError:
        pass

    # Integer subject IDs matching string subject IDs (e.g. 1 vs "1")
    df_type_mix = pd.DataFrame([
        {"subject_id": 1, "split": "train"},
        {"subject_id": "1", "split": "validation"},
    ])
    type_mix_caught = False
    try:
        verify_split_independence(df_type_mix, strict=True)
    except DataLeakageError:
        type_mix_caught = True
    assert type_mix_caught, "Integer vs string subject_id type mismatch allowed leakage bypass!"

    print("  -> Adversarial & large-scale split leakage detection: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING CHALLENGER M1 EMPIRICAL STRESS TEST SUITE")
    print("=" * 60)
    test_extreme_anisotropic_resampling()
    test_concentric_multiclass_mutual_exclusivity()
    test_micro_lesions_and_boundary_cases()
    test_inverse_spatial_mask_roundtrip()
    test_split_independence_adversarial_cases()
    print("=" * 60)
    print("ALL EMPIRICAL CHALLENGE ATTACK VECTORS DEFEATED / PASSED 100%!")
    print("=" * 60)
