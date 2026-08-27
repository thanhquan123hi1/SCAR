"""Anatomical Constraints and Clinical Consistency for Cardiac MRI Scar Segmentation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, label as nd_label

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
    tolerance_mm: float | None = 2.5,
    spacing: tuple[float, ...] | None = None,
    min_scar_voxels: int = 5,
    min_scar_volume_mm3: float | None = 15.0,
    cardiac_classes: tuple[int, ...] = (1, 2, 4),
) -> np.ndarray | torch.Tensor:
    """Enforce anatomical pathology rules on cardiac segmentation masks.
    
    Medical Pathology Rule:
    - Myocardial scar can ONLY exist within or directly contiguous with the myocardial wall or cardiac cavity.
    - Large transmural infarctions (full-wall scars on 2D/3D slices) are strictly preserved.
    - Spurious scar predictions floating in air, distant blood pools, or non-cardiac tissue are removed.
    
    Args:
        mask: Integer segmentation mask of shape (H, W) or (D, H, W).
        scar_class: Class ID of scar (default: 3).
        myo_class: Class ID of myocardium (default: 2).
        dilation_voxels: Fallback voxel expansion when spacing is not provided.
        tolerance_mm: Physical expansion tolerance in mm (default: 2.5 mm).
        spacing: Voxel spacing tuple (mm).
        min_scar_voxels: Fallback minimum connected voxel threshold when spacing is None.
        min_scar_volume_mm3: Physical minimum scar volume in mm³ (default: 15.0 mm³ ≈ 0.015 mL).
        cardiac_classes: Class IDs of cardiac anatomy anchors (default: (1, 2, 4) for LV, Myo, RV).
        
    Returns:
        Cleaned mask with anatomical consistency.
    """
    is_tensor = (torch is not None) and isinstance(mask, torch.Tensor)
    device = mask.device if is_tensor else None
    arr = mask.cpu().numpy() if is_tensor else np.asarray(mask).copy()

    if scar_class not in arr:
        return mask

    # 1. Extract Myocardium, Scar, and Cardiac Cavity (LV / RV blood pools)
    myo_mask = (arr == myo_class)
    scar_mask = (arr == scar_class)

    if not np.any(scar_mask):
        return mask

    cavity_classes = [c for c in cardiac_classes if c not in (scar_class, myo_class)]
    cavity_mask = np.isin(arr, cavity_classes) if cavity_classes else np.zeros_like(myo_mask, dtype=bool)

    # Cardiac anchor: Myocardium OR Cavity/Blood pool
    cardiac_anchor = myo_mask | cavity_mask

    # If there is no cardiac anatomy anywhere in the volume/image, scar has no anatomical anchor -> suppress
    if not np.any(cardiac_anchor):
        arr[scar_mask] = 0
        if is_tensor:
            return torch.from_numpy(arr).to(device=device, dtype=mask.dtype)
        return arr

    # Compute calibrated dilation iterations based on physical voxel spacing
    if spacing is not None and len(spacing) >= 2 and tolerance_mm is not None:
        min_inplane_spacing = max(1e-3, min(float(spacing[0]), float(spacing[1])))
        calibrated_iters = max(1, int(round(tolerance_mm / min_inplane_spacing)))
    elif spacing is not None and len(spacing) == 1 and tolerance_mm is not None:
        calibrated_iters = max(1, int(round(tolerance_mm / max(1e-3, float(spacing[0])))))
    else:
        calibrated_iters = max(1, dilation_voxels + 1)

    # Compute spacing-aware minimum scar voxel count from physical volume (fixes W5)
    if spacing is not None and min_scar_volume_mm3 is not None:
        voxel_vol_mm3 = float(np.prod(spacing[:arr.ndim]))
        effective_min_voxels = max(1, int(round(min_scar_volume_mm3 / max(1e-3, voxel_vol_mm3))))
    else:
        effective_min_voxels = min_scar_voxels

    # Full structuring element connectivity (3D 6-connectivity for 3D volumes, 2D 4-connectivity for 2D slices)
    # Preserves transmural apical scars connecting through-plane to myocardium on adjacent slices
    struct = generate_binary_structure(arr.ndim, 1)

    # Cardiac wall expansion zone (allows scar contiguous with or within the Myo ∪ Scar wall)
    cardiac_wall = myo_mask | scar_mask
    wall_zone = binary_dilation(cardiac_wall, structure=struct, iterations=calibrated_iters)

    # Dilated anatomical anchors (Myocardium and Cardiac Blood Pool)
    myo_dilated = binary_dilation(myo_mask, structure=struct, iterations=calibrated_iters) if np.any(myo_mask) else np.zeros_like(myo_mask, dtype=bool)
    cavity_dilated = binary_dilation(cavity_mask, structure=struct, iterations=calibrated_iters) if np.any(cavity_mask) else np.zeros_like(cavity_mask, dtype=bool)
    cardiac_dilated = myo_dilated | cavity_dilated

    # Process each connected component of scar independently
    labeled_scar, num_features = nd_label(scar_mask, structure=struct)
    for feat_id in range(1, num_features + 1):
        feat_mask = (labeled_scar == feat_id)
        feat_size = int(np.sum(feat_mask))

        # Check if scar component connects to cardiac anatomy (Myo wall or Cavity blood pool)
        connects_to_cardiac = bool(np.any(feat_mask & cardiac_dilated))

        if not connects_to_cardiac:
            # Isolated false-positive artifact (e.g. floating in air/blood pool/liver)
            arr[feat_mask] = 0
        elif feat_size < effective_min_voxels:
            # Tiny noisy speckle (< effective_min_voxels): revert to myocardium if within wall and myo exists, else 0
            if np.any(myo_mask):
                arr[feat_mask] = np.where(wall_zone[feat_mask], myo_class, 0)
            else:
                arr[feat_mask] = 0

    if is_tensor:
        return torch.from_numpy(arr).to(device=device, dtype=mask.dtype)
    return arr


# Alias for anatomical postprocessing
postprocess_predictions = enforce_anatomical_constraints
