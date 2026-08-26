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
    return np.logical_xor(
        mask,
        binary_erosion(mask, structure=structure, border_value=0),
    )


def hd95_binary(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...] | None = None,
) -> float | None:
    """Return physical-space 95th percentile Hausdorff Distance (HD95 in mm)."""
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)

    if not pred.any() and not truth.any():
        return None  # Not applicable (both empty)
    if not pred.any() or not truth.any():
        return float("inf")  # Complete miss

    sampling = spacing if spacing is not None else tuple([1.0] * pred.ndim)
    pred_surf = _surface(pred)
    truth_surf = _surface(truth)

    truth_dist = distance_transform_edt(~truth_surf, sampling=sampling)
    pred_dist = distance_transform_edt(~pred_surf, sampling=sampling)

    distances = np.concatenate([truth_dist[pred_surf], pred_dist[truth_surf]])
    return float(np.percentile(distances, 95))


def dice_score(pred: np.ndarray, target: np.ndarray, *, num_classes: int) -> dict[int, float]:
    """Per-class Dice score."""
    scores: dict[int, float] = {}
    for c in range(1, num_classes):
        p = pred == c
        t = target == c
        intersection = int((p & t).sum())
        denominator = int(p.sum() + t.sum())
        scores[c] = (2.0 * intersection / denominator) if denominator > 0 else (1.0 if not p.any() and not t.any() else 0.0)
    return scores


def iou_score(pred: np.ndarray, target: np.ndarray, *, num_classes: int) -> dict[int, float]:
    """Per-class IoU score."""
    scores: dict[int, float] = {}
    for c in range(1, num_classes):
        p = pred == c
        t = target == c
        intersection = int((p & t).sum())
        union = int((p | t).sum())
        scores[c] = (intersection / union) if union > 0 else (1.0 if not p.any() and not t.any() else 0.0)
    return scores


def mean_dice(scores: dict[int, float]) -> float:
    """Mean Dice across valid foreground classes."""
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
    """Calculate clinical LGE scar volume (mL) and myocardial scar mass (g)."""
    array = np.asarray(mask)
    scar_voxels = int(np.sum(array == scar_label))
    myocardium_voxels = int(np.sum(array == myocardium_label))

    voxel_volume_mm3 = float(np.prod(spacing))
    volume_ml = scar_voxels * voxel_volume_mm3 / 1000.0
    denominator = scar_voxels + myocardium_voxels

    return ScarMetrics(
        scar_volume_ml=float(volume_ml),
        scar_mass_g=float(volume_ml * tissue_density_g_per_ml),
        scar_voxels=scar_voxels,
        myocardium_voxels=myocardium_voxels,
        scar_fraction_of_myo_plus_scar=(
            None if denominator == 0 else float(scar_voxels / denominator)
        ),
    )
