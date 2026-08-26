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
    *,
    rare_classes: Sequence[int] = (3, 2),
    rare_boost: float = 5.0,
    foreground_boost: float = 1.8,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples slices/volumes containing rare classes.
    
    Args:
        dataset: Instance of LgeSaxDataset or LgeLaxDataset.
        rare_classes: Class IDs considered rare (e.g. [3, 2] for Scar and Myo).
        rare_boost: Sampling weight multiplier for samples containing primary rare class (e.g. 3).
        foreground_boost: Sampling weight multiplier for samples containing other foreground classes.
        
    Returns:
        WeightedRandomSampler configured for DataLoader.
    """
    weights: list[float] = []
    primary_rare = rare_classes[0] if rare_classes else 3
    secondary_rare = set(rare_classes[1:]) if len(rare_classes) > 1 else set()

    for idx in range(len(dataset)):
        weight = 1.0
        try:
            # Check LgeLaxDataset fast path
            if hasattr(dataset, "slices"):
                item = dataset.slices[idx]
                parent_id = str(item["row"].record_id)
                slice_idx = item["slice_idx"]
                cache_file = dataset.cache_dir / f"{parent_id}.npz" if getattr(dataset, "cache_dir", None) else None
                
                if cache_file and cache_file.exists():
                    with np.load(cache_file, allow_pickle=True) as data:
                        if "label" in data:
                            lbl_slice = data["label"][slice_idx]
                            unique_cls = np.unique(lbl_slice)
                            if primary_rare in unique_cls:
                                weight = rare_boost
                            elif any(c in unique_cls for c in secondary_rare) or np.any(unique_cls > 0):
                                weight = foreground_boost
                elif hasattr(dataset, "data_root"):
                    row = item["row"]
                    label_path = getattr(row, "label_path", None)
                    if label_path and (dataset.data_root / str(label_path)).exists():
                        raw_lbl = nib.as_closest_canonical(nib.load(str(dataset.data_root / str(label_path))))
                        lbl_data = np.asanyarray(raw_lbl.dataobj)
                        lbl_slice = lbl_data[:, :, slice_idx] if lbl_data.ndim >= 3 else lbl_data
                        unique_cls = np.unique(lbl_slice)
                        if primary_rare in unique_cls:
                            weight = rare_boost
                        elif any(c in unique_cls for c in secondary_rare) or np.any(unique_cls > 0):
                            weight = foreground_boost
                    
            # Check LgeSaxDataset fast path
            elif hasattr(dataset, "records"):
                row = dataset.records.iloc[idx]
                rec_id = str(row.get("record_id", f"rec_{idx}"))
                cache_file = dataset.cache_dir / f"{rec_id}.npz" if getattr(dataset, "cache_dir", None) else None
                
                if cache_file and cache_file.exists():
                    with np.load(cache_file, allow_pickle=True) as data:
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
        logger.warning(
            "⚠️ ALL %d samples have weight=1.0 — rare-class sampler is NOT active! "
            "This likely means cache files are missing. Run process_and_save.py first.",
            len(weights),
        )

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
