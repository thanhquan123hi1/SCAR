"""Weighted Sampler for rare medical classes (Scar / Myocardium)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


def build_rare_class_sampler(
    dataset: Dataset,
    rare_classes: list[int] | None = None,
    rare_boost: float = 2.5,
    foreground_boost: float = 1.5,
    strict: bool = False,
) -> WeightedRandomSampler | None:
    """Build a WeightedRandomSampler that over-samples records containing rare pathology.
    
    If no samples can be boosted (all weights=1.0 due to missing labels/cache), returns None
    so DataLoader safely falls back to standard epoch shuffling without replacement.
    """
    if rare_classes is None:
        rare_classes = [3, 2]  # Default: [Scar, Myocardium]

    primary_rare = rare_classes[0] if rare_classes else 3
    secondary_rare = rare_classes[1:] if len(rare_classes) > 1 else []

    weights: list[float] = []

    for idx in range(len(dataset)):
        weight = 1.0
        try:
            if hasattr(dataset, "slices"):
                item = dataset.slices[idx]
                row = item["row"]
                cache_dir = getattr(dataset, "cache_dir", None)
                slice_idx = item.get("slice_idx", 0)

                if cache_dir and (cache_dir / f"{row.record_id}.npz").exists():
                    with np.load(cache_dir / f"{row.record_id}.npz", allow_pickle=True) as data:
                        if "label" in data:
                            lbl = data["label"]
                            # Check slice or volume
                            lbl_slice = lbl[slice_idx] if lbl.ndim >= 3 and slice_idx < lbl.shape[0] else lbl
                            unique_cls = np.unique(lbl_slice)
                            if primary_rare in unique_cls:
                                weight = rare_boost
                            elif any(c in unique_cls for c in secondary_rare) or np.any(unique_cls > 0):
                                weight = foreground_boost
                elif hasattr(dataset, "data_root"):
                    label_path = row.get("label_path", None)
                    if label_path and (dataset.data_root / str(label_path)).exists():
                        raw_lbl = nib.as_closest_canonical(nib.load(str(dataset.data_root / str(label_path))))
                        data_arr = np.asanyarray(raw_lbl.dataobj)
                        lbl_slice = data_arr[:, :, slice_idx] if data_arr.ndim >= 3 and slice_idx < data_arr.shape[2] else data_arr
                        unique_cls = np.unique(lbl_slice)
                        if primary_rare in unique_cls:
                            weight = rare_boost
                        elif any(c in unique_cls for c in secondary_rare) or np.any(unique_cls > 0):
                            weight = foreground_boost
            elif hasattr(dataset, "records"):
                row = dataset.records.iloc[idx]
                cache_dir = getattr(dataset, "cache_dir", None)

                if cache_dir and (cache_dir / f"{row.record_id}.npz").exists():
                    with np.load(cache_dir / f"{row.record_id}.npz", allow_pickle=True) as data:
                        if "label" in data:
                            unique_cls = np.unique(data["label"])
                            if primary_rare in unique_cls:
                                weight = rare_boost
                            elif any(c in unique_cls for c in secondary_rare) or np.any(unique_cls > 0):
                                weight = foreground_boost
                elif hasattr(dataset, "data_root"):
                    label_path = row.get("label_path", None)
                    if label_path and (dataset.data_root / str(label_path)).exists():
                        raw_lbl = nib.as_closest_canonical(nib.load(str(dataset.data_root / str(label_path))))
                        unique_cls = np.unique(raw_lbl.dataobj)
                        if primary_rare in unique_cls:
                            weight = rare_boost
                        elif any(c in unique_cls for c in secondary_rare) or np.any(unique_cls > 0):
                            weight = foreground_boost
        except Exception as e:
            logger.warning(
                "Failed to compute sample weight for index %d: %s.",
                idx, e,
            )
            weight = 1.0

        weights.append(weight)

    # Validate that sampler is actually boosting some samples
    num_default = sum(1 for w in weights if w == 1.0)
    if num_default == len(weights) and len(weights) > 0:
        if strict:
            raise RuntimeError(
                f"All {len(weights)} samples have default weight=1.0. "
                "Rare-class sampler cannot boost any samples because labels/cache are missing."
            )
        logger.warning(
            "⚠️ ALL %d samples have weight=1.0 — rare-class sampler cannot boost any samples! "
            "Falling back to standard DataLoader shuffle=True to avoid degenerate replacement sampling.",
            len(weights),
        )
        return None

    weights_tensor = torch.tensor(weights, dtype=torch.double)
    num_boosted = sum(1 for w in weights if w >= rare_boost)
    logger.info(
        f"Built WeightedRareClassSampler: {len(weights)} items, {num_boosted} with primary rare boost (weight={rare_boost})."
    )

    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(weights_tensor),
        replacement=True,
    )
