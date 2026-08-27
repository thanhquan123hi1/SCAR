"""Medical segmentation metrics and clinical LGE scar quantification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


@dataclass(frozen=True)
class ScarMetrics:
    """LGE SAX scar volume, mass, and percentage summary."""

    scar_volume_ml: float
    scar_mass_g: float
    scar_voxels: int
    myocardium_voxels: int
    scar_fraction_of_myo_plus_scar: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _surface(mask: np.ndarray) -> np.ndarray:
    structure = generate_binary_structure(mask.ndim, 1)
    surf = np.logical_xor(
        mask,
        binary_erosion(mask, structure=structure, border_value=0),
    )
    # Robust guard (M6): if erosion eliminated small structure, treat entire mask as surface
    if not np.any(surf) and np.any(mask):
        return mask.copy()
    return surf


def compute_fov_diagonal(
    shape: tuple[int, ...],
    spacing: tuple[float, ...] | None = None,
) -> float:
    """Compute physical Field-of-View (FOV) diagonal length in mm: sqrt(sum((shape_i * spacing_i)^2))."""
    if spacing is not None:
        if len(spacing) < len(shape):
            sp = tuple(float(s) for s in spacing) + tuple([1.0] * (len(shape) - len(spacing)))
        else:
            sp = tuple(float(s) for s in spacing[:len(shape)])
    else:
        sp = tuple([1.0] * len(shape))
    return float(np.sqrt(sum((float(dim) * float(s)) ** 2 for dim, s in zip(shape, sp))))


def hd95_binary(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    penalty_distance: float | None = None,
    empty_value: float | None = 0.0,
) -> float | None:
    """Return physical-space 95th percentile Hausdorff Distance (HD95 in mm).
    
    Aligned with Metrics Reloaded (Nature Methods 2024):
    - If both prediction and target are empty (Symmetric True Negative): returns empty_value (default: 0.0 mm).
    - If one is empty and the other is not (complete miss / false alarm): returns dynamic FOV diagonal
      sqrt(sum((dim * sp)^2)) when penalty_distance is None, or penalty_distance when explicitly provided.
    - If both are non-empty: computes the 95th percentile symmetric surface distance in mm.
    """
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)

    if not pred.any() and not truth.any():
        return empty_value  # Symmetrically handled True Negative (both empty -> 0.0 mm)

    if spacing is not None:
        if len(spacing) < pred.ndim:
            sampling = tuple(float(s) for s in spacing) + tuple([1.0] * (pred.ndim - len(spacing)))
        else:
            sampling = tuple(float(s) for s in spacing[:pred.ndim])
    else:
        sampling = tuple([1.0] * pred.ndim)

    # Dynamic patient FOV diagonal scaling when penalty_distance is None
    if penalty_distance is None:
        penalty_val = compute_fov_diagonal(pred.shape, sampling)
    else:
        penalty_val = float(penalty_distance)

    if not pred.any() or not truth.any():
        return penalty_val

    pred_surf = _surface(pred)
    truth_surf = _surface(truth)

    if not pred_surf.any() or not truth_surf.any():
        return penalty_val

    truth_dist = distance_transform_edt(~truth_surf, sampling=sampling)
    pred_dist = distance_transform_edt(~pred_surf, sampling=sampling)

    distances = np.concatenate([truth_dist[pred_surf], pred_dist[truth_surf]])
    if distances.size == 0:
        return penalty_val
    return float(np.percentile(distances, 95))


def dice_score_symmetric(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    num_classes: int,
) -> dict[int, float]:
    """Per-class Dice score with symmetric True Negative handling (empty GT & empty Pred -> 1.0)."""
    return dice_score(pred, target, num_classes=num_classes, empty_value=1.0)


def iou_score_symmetric(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    num_classes: int,
) -> dict[int, float]:
    """Per-class IoU score with symmetric True Negative handling (empty GT & empty Pred -> 1.0)."""
    return iou_score(pred, target, num_classes=num_classes, empty_value=1.0)


def dice_score(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    num_classes: int,
    empty_value: float = float("nan"),
) -> dict[int, float]:
    """Per-class Dice score.
    
    When both pred and target are empty: returns empty_value (default NaN)
    to avoid artificial score inflation.
    """
    scores: dict[int, float] = {}
    for c in range(1, num_classes):
        p = pred == c
        t = target == c
        intersection = int((p & t).sum())
        denominator = int(p.sum() + t.sum())
        if denominator > 0:
            scores[c] = float(2.0 * intersection / denominator)
        else:
            scores[c] = empty_value
    return scores


def iou_score(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    num_classes: int,
    empty_value: float = float("nan"),
) -> dict[int, float]:
    """Per-class IoU score."""
    scores: dict[int, float] = {}
    for c in range(1, num_classes):
        p = pred == c
        t = target == c
        intersection = int((p & t).sum())
        union = int((p | t).sum())
        if union > 0:
            scores[c] = float(intersection / union)
        else:
            scores[c] = empty_value
    return scores


def mean_dice(scores: dict[int, float]) -> float:
    """Mean Dice across valid foreground classes (ignoring NaNs)."""
    valid = [v for v in scores.values() if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def calculate_scar_metrics(
    mask: np.ndarray,
    *,
    spacing: tuple[float, ...],
    scar_label: int = 3,
    myocardium_label: int = 2,
    tissue_density_g_per_ml: float = 1.05,
) -> ScarMetrics:
    """Calculate clinical LGE scar volume (mL) and myocardial scar mass (g).
    
    Volume and mass are only clinically defined for 3D SAX stacks. For 2D single slices,
    NaN is returned to prevent physical dimensional errors.
    """
    array = np.asarray(mask)
    scar_voxels = int(np.sum(array == scar_label))
    myocardium_voxels = int(np.sum(array == myocardium_label))

    is_3d = array.ndim == 3 and len(spacing) >= 3
    if is_3d:
        voxel_volume_mm3 = float(np.prod(spacing[:3]))
        volume_ml = scar_voxels * voxel_volume_mm3 / 1000.0
        mass_g = volume_ml * tissue_density_g_per_ml
    else:
        volume_ml = float("nan")
        mass_g = float("nan")

    denominator = scar_voxels + myocardium_voxels

    return ScarMetrics(
        scar_volume_ml=float(volume_ml),
        scar_mass_g=float(mass_g),
        scar_voxels=scar_voxels,
        myocardium_voxels=myocardium_voxels,
        scar_fraction_of_myo_plus_scar=(
            None if denominator == 0 else float(scar_voxels / denominator)
        ),
    )
