"""Anatomical Constraints and Clinical Consistency for Cardiac MRI Scar Segmentation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, label as nd_label

try:
    import torch
except ImportError:
    torch = None


def enforce_anatomical_constraints(
    mask: np.ndarray | torch.Tensor,
    *,
    scar_class: int = 3,
    myo_class: int = 2,
    dilation_voxels: int = 1,
    min_scar_voxels: int = 5,
) -> np.ndarray | torch.Tensor:
    """Enforce anatomical pathology rules on cardiac segmentation masks.
    
    Medical Rule:
    Myocardial scar can ONLY exist within or directly abutting the myocardial wall (Class 2).
    Spurious scar predictions floating in blood pools (LV/RV cavity) or outside the heart are removed.
    
    Args:
        mask: Integer segmentation mask of shape (H, W) or (D, H, W).
        scar_class: Class ID of scar (default: 3).
        myo_class: Class ID of myocardium (default: 2).
        dilation_voxels: Voxel expansion for myocardium border tolerance.
        min_scar_voxels: Minimum connected volume threshold (removes isolated noise).
        
    Returns:
        Cleaned mask with anatomical consistency.
    """
    is_tensor = (torch is not None) and isinstance(mask, torch.Tensor)
    device = mask.device if is_tensor else None
    arr = mask.cpu().numpy() if is_tensor else np.asarray(mask).copy()

    if scar_class not in arr:
        return mask

    # 1. Extract Myocardium mask (Grounding tissue)
    myo_mask = (arr == myo_class)
    scar_mask = (arr == scar_class)

    if not np.any(scar_mask):
        return mask

    # 2. Derive valid anatomical zone from Myocardium
    struct = np.ones((3,) * mask.ndim, dtype=bool)
    if np.any(myo_mask):
        # Dilation allows scar within or immediately adjacent (subendo/subepi) to myocardium
        valid_zone = binary_dilation(myo_mask, structure=struct, iterations=max(1, dilation_voxels + 3))
    else:
        valid_zone = np.zeros_like(scar_mask)

    # 3. Suppress spurious scar predictions outside the valid myocardial zone
    invalid_scar = scar_mask & (~valid_zone)
    arr[invalid_scar] = 0

    # 4. Remove isolated noisy micro-components (< min_scar_voxels)
    cleaned_scar = (arr == scar_class)
    if min_scar_voxels > 1 and np.any(cleaned_scar):
        labeled_scar, num_features = nd_label(cleaned_scar)
        for feat_id in range(1, num_features + 1):
            feat_mask = (labeled_scar == feat_id)
            if np.sum(feat_mask) < min_scar_voxels:
                # Revert small noisy scar to normal myocardium only if within myocardial zone, else 0
                arr[feat_mask] = np.where(valid_zone[feat_mask], myo_class, 0)

    if is_tensor:
        return torch.from_numpy(arr).to(device=device, dtype=mask.dtype)
    return arr
