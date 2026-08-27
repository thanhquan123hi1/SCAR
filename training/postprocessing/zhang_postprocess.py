"""Post-processing module for Zhang (2021) segmentation method.

Reference:
    - Lalande, A. et al. (2022). Section 3.1.5 Post-processing:
      "Zhang adopted another simple treatment that removed all the scattered pixels
      from the segmentation."
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
from scipy import ndimage


def remove_scattered_pixels(
    mask: np.ndarray,
    target_classes: Optional[Tuple[int, ...]] = (2, 3, 4),
    min_size_voxels: int = 10,
) -> np.ndarray:
    """Removes scattered / isolated small pixel clusters from specific target classes.

    Args:
        mask: 2D or 3D integer array of segmentation class IDs.
        target_classes: Tuple of class IDs to apply component filtering to (e.g. Scar, Myocardium).
        min_size_voxels: Minimum number of contiguous voxels required to keep a component.
    Returns:
        Cleaned integer mask with scattered small clusters removed (reverted to background or nearest).
    """
    cleaned = mask.copy()
    structure = ndimage.generate_binary_structure(mask.ndim, 1)  # 4-connectivity (2D) or 6-connectivity (3D)

    if target_classes is None:
        target_classes = tuple(c for c in np.unique(mask) if c != 0)

    for c in target_classes:
        binary_c = (mask == c)
        if not np.any(binary_c):
            continue

        labeled_array, num_features = ndimage.label(binary_c, structure=structure)
        if num_features == 0:
            continue

        component_sizes = ndimage.sum(binary_c, labeled_array, range(1, num_features + 1))
        # Find components smaller than threshold
        too_small = np.where(component_sizes < min_size_voxels)[0] + 1
        if len(too_small) > 0:
            remove_mask = np.isin(labeled_array, too_small)
            cleaned[remove_mask] = 0

    return cleaned


def anatomical_clean_scar(
    mask: np.ndarray,
    scar_class: int = 3,
    myo_class: int = 2,
    max_distance_voxels: float = 3.0,
) -> np.ndarray:
    """Enforces anatomical constraint that scar must be located inside or adjacent to myocardium.

    Args:
        mask: 2D or 3D integer segmentation mask.
        scar_class: Label ID for Scar / Infarction.
        myo_class: Label ID for Myocardium.
        max_distance_voxels: Maximum allowed distance from myocardium before being pruned.
    """
    cleaned = mask.copy()
    scar_mask = (mask == scar_class)
    myo_mask = (mask == myo_class)

    if not np.any(scar_mask) or not np.any(myo_mask):
        return cleaned

    # Distance transform from myocardium
    dist_from_myo = ndimage.distance_transform_edt(~myo_mask)
    isolated_scar = scar_mask & (dist_from_myo > max_distance_voxels)

    if np.any(isolated_scar):
        cleaned[isolated_scar] = 0

    return cleaned


def zhang_postprocess(
    mask: np.ndarray,
    min_size_voxels: int = 8,
    scar_class: int = 3,
    myo_class: int = 2,
    enforce_myo_proximity: bool = True,
) -> np.ndarray:
    """Full Zhang (2021) post-processing pipeline.

    1. Removes scattered isolated pixels.
    2. Enforces anatomical proximity between scar and myocardium.
    """
    out = remove_scattered_pixels(mask, min_size_voxels=min_size_voxels)
    if enforce_myo_proximity and scar_class in out and myo_class in out:
        out = anatomical_clean_scar(out, scar_class=scar_class, myo_class=myo_class)
    return out
