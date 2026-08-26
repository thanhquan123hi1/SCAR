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
    
    Medical Pathology Rule:
    - Myocardial scar can ONLY exist within or directly contiguous with the myocardial wall.
    - Large transmural infarctions (full-wall scars) are strictly preserved.
    - Spurious scar predictions floating in blood pools or outside the heart are removed.
    
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

    # 1. Extract Myocardium and Scar masks
    myo_mask = (arr == myo_class)
    scar_mask = (arr == scar_class)

    if not np.any(scar_mask):
        return mask

    # If myocardium is present, define connected myocardial wall region
    if np.any(myo_mask):
        # Use in-plane structuring element for 3D stacks to avoid anisotropic Z bleeding
        if arr.ndim == 3:
            # (D, H, W) layout: expand in H, W (axes 1, 2)
            struct = np.zeros((3, 3, 3), dtype=bool)
            struct[1, :, :] = True
        else:
            struct = np.ones((3, 3), dtype=bool)

        # Cardiac wall expansion zone (allows scar contiguous with or within the wall)
        wall_zone = binary_dilation(myo_mask | scar_mask, structure=struct, iterations=max(1, dilation_voxels))
        myo_dilated = binary_dilation(myo_mask, structure=struct, iterations=max(2, dilation_voxels + 4))

        # 2. Process each connected component of scar independently
        labeled_scar, num_features = nd_label(scar_mask)
        for feat_id in range(1, num_features + 1):
            feat_mask = (labeled_scar == feat_id)
            feat_size = np.sum(feat_mask)

            # Check if this scar component connects to the myocardial wall
            connects_to_myo = np.any(feat_mask & myo_dilated)

            if not connects_to_myo:
                # Isolated false-positive artifact (e.g. floating in air/blood pool/liver)
                arr[feat_mask] = 0
            elif feat_size < min_scar_voxels:
                # Tiny noisy speckle (< min_scar_voxels): revert to myocardium if within wall, else 0
                arr[feat_mask] = np.where(wall_zone[feat_mask], myo_class, 0)

    if is_tensor:
        return torch.from_numpy(arr).to(device=device, dtype=mask.dtype)
    return arr
