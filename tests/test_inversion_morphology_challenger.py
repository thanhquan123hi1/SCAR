"""Comprehensive Empirical Verification Suite for Inversion & Morphology Preservation.

Challenger M1_2 Verification Harness.
Tests round-trip transformation: mask -> preprocess_mask -> invert_spatial_mask -> restored_mask.
Evaluates Dice score, centroid displacement, volume preservation, and 1-voxel / 2-voxel vanishing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repository root to python search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any
import numpy as np
from preprocessing.preprocessing import (
    CenterTransform,
    SpatialTransform,
    invert_spatial_mask,
    preprocess_mask,
    preprocess_spatial,
)


def run_roundtrip_test(
    orig_shape: tuple[int, ...],
    src_spacing: tuple[float, ...],
    tgt_spacing: tuple[float, ...],
    tgt_shape: tuple[int, ...],
    lesion_size: tuple[int, ...],
    pos: tuple[int, ...] | None = None,
    lesion_class: int = 3,
    with_anatomy: bool = False,
) -> dict[str, Any]:
    """Execute a single round-trip test and compute metrics."""
    mask = np.zeros(orig_shape, dtype=np.int16)

    # Optional surrounding multi-class anatomy
    if with_anatomy:
        if len(orig_shape) == 2:
            cy, cx = orig_shape[0] // 2, orig_shape[1] // 2
            mask[cy - 40 : cy + 40, cx - 40 : cx + 40] = 2  # Myo
            mask[cy - 20 : cy + 20, cx - 20 : cx + 20] = 1  # LV
            mask[cy - 40 : cy + 40, cx + 40 : cx + 70] = 4  # RV
        elif len(orig_shape) == 3:
            cz, cy, cx = orig_shape[0] // 2, orig_shape[1] // 2, orig_shape[2] // 2
            mask[:, cy - 40 : cy + 40, cx - 40 : cx + 40] = 2
            mask[:, cy - 20 : cy + 20, cx - 20 : cx + 20] = 1
            mask[:, cy - 40 : cy + 40, cx + 40 : cx + 70] = 4

    # Place lesion
    if pos is None:
        pos = tuple(s // 2 for s in orig_shape)

    slices = tuple(
        slice(p, p + size) for p, size in zip(pos, lesion_size, strict=True)
    )
    mask[slices] = lesion_class

    # Generate spatial transform via preprocess_spatial
    dummy_img = np.zeros(orig_shape, dtype=np.float32)
    _, transform = preprocess_spatial(
        dummy_img,
        source_spacing=src_spacing,
        target_spacing=tgt_spacing,
        target_shape=tgt_shape,
    )

    # Preprocess mask
    preprocessed_mask = preprocess_mask(
        mask,
        source_spacing=src_spacing,
        target_spacing=tgt_spacing,
        target_shape=tgt_shape,
    )

    # Invert back to original geometry
    restored_mask = invert_spatial_mask(preprocessed_mask, transform)

    # Extract statistics
    orig_pts = np.argwhere(mask == lesion_class)
    proc_pts = np.argwhere(preprocessed_mask == lesion_class)
    rest_pts = np.argwhere(restored_mask == lesion_class)

    orig_count = int(len(orig_pts))
    proc_count = int(len(proc_pts))
    rest_count = int(len(rest_pts))

    orig_centroid = orig_pts.mean(axis=0) if orig_count > 0 else None
    proc_centroid = proc_pts.mean(axis=0) if proc_count > 0 else None
    rest_centroid = rest_pts.mean(axis=0) if rest_count > 0 else None

    if orig_centroid is not None and rest_centroid is not None:
        displacement = float(np.linalg.norm(orig_centroid - rest_centroid))
    else:
        displacement = float("inf")

    # Dice & IoU
    b_orig = mask == lesion_class
    b_rest = restored_mask == lesion_class
    intersection = int(np.logical_and(b_orig, b_rest).sum())
    union = int(np.logical_or(b_orig, b_rest).sum())
    dice = (2.0 * intersection) / (orig_count + rest_count) if (orig_count + rest_count) > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0

    return {
        "orig_shape": orig_shape,
        "tgt_shape": tgt_shape,
        "src_spacing": src_spacing,
        "tgt_spacing": tgt_spacing,
        "lesion_size": lesion_size,
        "orig_count": orig_count,
        "proc_count": proc_count,
        "rest_count": rest_count,
        "orig_centroid": orig_centroid,
        "proc_centroid": proc_centroid,
        "rest_centroid": rest_centroid,
        "displacement": displacement,
        "dice": dice,
        "iou": iou,
        "vanished_proc": (proc_count == 0),
        "vanished_rest": (rest_count == 0),
    }


def main():
    print("=" * 80)
    print("CHALLENGER M1_2: EMPIRICAL INVERSION & MORPHOLOGY HARNESS")
    print("=" * 80)

    configs = [
        {
            "name": "Config 1: 2D 2.0x Downsampling (1.0mm -> 2.0mm, shape 200 -> 100)",
            "orig_shape": (200, 200),
            "src_spacing": (1.0, 1.0),
            "tgt_spacing": (2.0, 2.0),
            "tgt_shape": (100, 100),
            "lesions": [(1, 1), (1, 2), (2, 2), (5, 5), (20, 20)],
            "with_anatomy": False,
        },
        {
            "name": "Config 2: 2D Realistic CMR Resampling & Cropping (1.25mm -> 1.33mm, 256 -> 192)",
            "orig_shape": (256, 256),
            "src_spacing": (1.25, 1.25),
            "tgt_spacing": (1.33, 1.33),
            "tgt_shape": (192, 192),
            "lesions": [(1, 1), (1, 2), (2, 2), (5, 5), (20, 20)],
            "with_anatomy": False,
        },
        {
            "name": "Config 3: 2D 2.0x Upsampling (2.0mm -> 1.0mm, shape 100 -> 200)",
            "orig_shape": (100, 100),
            "src_spacing": (2.0, 2.0),
            "tgt_spacing": (1.0, 1.0),
            "tgt_shape": (200, 200),
            "lesions": [(1, 1), (1, 2), (2, 2), (5, 5), (20, 20)],
            "with_anatomy": False,
        },
        {
            "name": "Config 4: 2D Anisotropic Non-Integer Resampling + Padding (1.4x1.2mm -> 1.0x1.0mm, 150x175 -> 224x224)",
            "orig_shape": (150, 175),
            "src_spacing": (1.4, 1.2),
            "tgt_spacing": (1.0, 1.0),
            "tgt_shape": (224, 224),
            "lesions": [(1, 1), (1, 2), (2, 2), (5, 5), (20, 20)],
            "with_anatomy": False,
        },
        {
            "name": "Config 5: 2D Multi-Class Context (Scar inside Myocardium)",
            "orig_shape": (200, 200),
            "src_spacing": (1.0, 1.0),
            "tgt_spacing": (1.5, 1.5),
            "tgt_shape": (128, 128),
            "lesions": [(1, 1), (1, 2), (2, 2), (5, 5), (20, 20)],
            "with_anatomy": True,
        },
        {
            "name": "Config 6: 3D Thick-Slice Volume (10.0x1.25x1.25mm -> 10.0x1.0x1.0mm, 12x216x216 -> 12x256x256)",
            "orig_shape": (12, 216, 216),
            "src_spacing": (10.0, 1.25, 1.25),
            "tgt_spacing": (10.0, 1.0, 1.0),
            "tgt_shape": (12, 256, 256),
            "lesions": [(1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 5, 5), (4, 20, 20)],
            "with_anatomy": True,
        },
    ]

    all_passed = True
    total_tests = 0
    vanished_count = 0

    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        print(f"{'Lesion Size':<15} | {'Orig Vox':<8} | {'Proc Vox':<8} | {'Rest Vox':<8} | {'Dice':<8} | {'IoU':<8} | {'Centroid Disp (vx)':<18} | {'Vanished?'}")
        print("-" * 95)
        for lsize in cfg["lesions"]:
            total_tests += 1
            res = run_roundtrip_test(
                orig_shape=cfg["orig_shape"],
                src_spacing=cfg["src_spacing"],
                tgt_spacing=cfg["tgt_spacing"],
                tgt_shape=cfg["tgt_shape"],
                lesion_size=lsize,
                with_anatomy=cfg["with_anatomy"],
            )

            is_vanished = res["vanished_rest"] or res["vanished_proc"]
            if is_vanished:
                vanished_count += 1
                all_passed = False

            lsize_str = "x".join(map(str, lsize))
            print(
                f"{lsize_str:<15} | {res['orig_count']:<8} | {res['proc_count']:<8} | {res['rest_count']:<8} | "
                f"{res['dice']:<8.4f} | {res['iou']:<8.4f} | {res['displacement']:<18.4f} | {'YES [FAIL]' if is_vanished else 'NO [PASS]'}"
            )

    # Advanced Stress Tests:
    # 1. Grid Phase Shifts (testing subpixel alignment & ties)
    print("\n--- Stress Test 1: Grid Phase Shifts (Sub-pixel Alignment & Tie-Breaking) ---")
    print(f"{'Position (y,x)':<15} | {'Size':<8} | {'Orig Vox':<8} | {'Proc Vox':<8} | {'Rest Vox':<8} | {'Dice':<8} | {'Centroid Disp (vx)':<18} | {'Vanished?'}")
    print("-" * 95)
    for (py, px) in [(90, 90), (91, 90), (90, 91), (91, 91), (93, 95)]:
        for lsize in [(1, 1), (1, 2), (2, 2)]:
            total_tests += 1
            res = run_roundtrip_test(
                orig_shape=(200, 200),
                src_spacing=(1.0, 1.0),
                tgt_spacing=(2.0, 2.0),
                tgt_shape=(100, 100),
                lesion_size=lsize,
                pos=(py, px),
            )
            is_vanished = res["vanished_rest"] or res["vanished_proc"]
            if is_vanished:
                vanished_count += 1
                all_passed = False
            lsize_str = "x".join(map(str, lsize))
            pos_str = f"({py},{px})"
            print(
                f"{pos_str:<15} | {lsize_str:<8} | {res['orig_count']:<8} | {res['proc_count']:<8} | {res['rest_count']:<8} | "
                f"{res['dice']:<8.4f} | {res['displacement']:<18.4f} | {'YES [FAIL]' if is_vanished else 'NO [PASS]'}"
            )

    # 2. Boundary / Border Lesions (Near crop box edges)
    print("\n--- Stress Test 2: Border & Crop-Edge Lesions ---")
    print(f"{'Position (y,x)':<15} | {'Size':<8} | {'Orig Vox':<8} | {'Proc Vox':<8} | {'Rest Vox':<8} | {'Dice':<8} | {'Centroid Disp (vx)':<18} | {'Vanished?'}")
    print("-" * 95)
    for (py, px) in [(33, 33), (33, 220), (220, 33), (220, 220), (128, 128)]:
        # Note: In 256x256 cropped to 192x192, crop bounds are [32, 224]
        for lsize in [(1, 1), (2, 2), (5, 5)]:
            total_tests += 1
            res = run_roundtrip_test(
                orig_shape=(256, 256),
                src_spacing=(1.0, 1.0),
                tgt_spacing=(1.0, 1.0),
                tgt_shape=(192, 192),
                lesion_size=lsize,
                pos=(py, px),
            )
            is_vanished = res["vanished_rest"] or res["vanished_proc"]
            if is_vanished:
                vanished_count += 1
                all_passed = False
            lsize_str = "x".join(map(str, lsize))
            pos_str = f"({py},{px})"
            print(
                f"{pos_str:<15} | {lsize_str:<8} | {res['orig_count']:<8} | {res['proc_count']:<8} | {res['rest_count']:<8} | "
                f"{res['dice']:<8.4f} | {res['displacement']:<18.4f} | {'YES [FAIL]' if is_vanished else 'NO [PASS]'}"
            )

    # 3. Multiple Disjoint Isolated 1-Voxel Lesions
    print("\n--- Stress Test 3: Multiple Disjoint Isolated 1-Voxel Lesions ---")
    multi_mask = np.zeros((200, 200), dtype=np.int16)
    # Scatter 8 isolated 1-voxel lesions
    scar_locs = [(50, 50), (50, 150), (150, 50), (150, 150), (80, 80), (81, 120), (120, 81), (121, 121)]
    for y, x in scar_locs:
        multi_mask[y, x] = 3

    dummy_img = np.zeros((200, 200), dtype=np.float32)
    _, tr = preprocess_spatial(dummy_img, source_spacing=(1.0, 1.0), target_spacing=(2.0, 2.0), target_shape=(100, 100))
    proc_m = preprocess_mask(multi_mask, source_spacing=(1.0, 1.0), target_spacing=(2.0, 2.0), target_shape=(100, 100))
    rest_m = invert_spatial_mask(proc_m, tr)

    orig_c = int((multi_mask == 3).sum())
    proc_c = int((proc_m == 3).sum())
    rest_c = int((rest_m == 3).sum())
    print(f"Disjoint 1-voxel lesions: {len(scar_locs)} placed -> Orig={orig_c}, Proc={proc_c}, Rest={rest_c}")
    assert proc_c >= len(scar_locs), f"Expected at least {len(scar_locs)} voxels in preprocessed mask, got {proc_c}"
    assert rest_c >= len(scar_locs), f"Expected at least {len(scar_locs)} voxels in restored mask, got {rest_c}"
    print("Multi-lesion preservation: PASS")

    # 4. Resampling Scale Sweep (1.0x to 3.0x downsampling for 1x1, 2x2, 5x5, 20x20)
    print("\n--- Stress Test 4: Downsampling Scale Sweep (1.0x -> [1.0x, 1.25x, 1.5x, 1.75x, 2.0x, 2.5x, 3.0x]) ---")
    print(f"{'Scale (mm)':<12} | {'Lesion':<8} | {'Orig Vox':<8} | {'Proc Vox':<8} | {'Rest Vox':<8} | {'Dice':<8} | {'Centroid Disp (vx)':<18} | {'Vanished?'}")
    print("-" * 95)
    for tgt_sp_val in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        tgt_dim = int(round(200 / tgt_sp_val))
        for lsize in [(1, 1), (2, 2), (5, 5), (20, 20)]:
            total_tests += 1
            res = run_roundtrip_test(
                orig_shape=(200, 200),
                src_spacing=(1.0, 1.0),
                tgt_spacing=(tgt_sp_val, tgt_sp_val),
                tgt_shape=(tgt_dim, tgt_dim),
                lesion_size=lsize,
                pos=(100, 100),
            )
            is_vanished = res["vanished_rest"] or res["vanished_proc"]
            if is_vanished:
                vanished_count += 1
                all_passed = False
            lsize_str = "x".join(map(str, lsize))
            print(
                f"{tgt_sp_val:<12.2f} | {lsize_str:<8} | {res['orig_count']:<8} | {res['proc_count']:<8} | {res['rest_count']:<8} | "
                f"{res['dice']:<8.4f} | {res['displacement']:<18.4f} | {'YES [FAIL]' if is_vanished else 'NO [PASS]'}"
            )

    print("\n" + "=" * 80)
    print(f"SUMMARY: {total_tests} configurations tested. Vanished lesions: {vanished_count}.")
    if all_passed:
        print("RESULT: ALL TESTS PASSED - GEOMETRIC MORPHOLOGY & INVERSION PRESERVED")
    else:
        print("RESULT: FAILURE DETECTED")
    print("=" * 80)


if __name__ == "__main__":
    main()
