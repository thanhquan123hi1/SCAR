"""Optional, explicit slice sampling for canonical MyoPS targets."""
from __future__ import annotations
import math
import torch
from torch.utils.data import WeightedRandomSampler


def build_rare_class_sampler(dataset, rare_classes=(2, 3), rare_boost=2.0,
                             foreground_boost=1.3, generator=None):
    """Inspect unaugmented canonical labels; do not consume augmentation RNG.

    This changes the training distribution and is disabled by default. Sampling
    uses replacement and keeps one epoch equal to the original number of slices.
    """
    if dataset.is_volume or len(dataset) < 1:
        raise ValueError("Rare-class sampling requires a nonempty slice dataset")
    if not rare_classes or any(c not in (1, 2, 3) for c in rare_classes):
        raise ValueError("rare_classes must contain canonical foreground class IDs")
    if any(not math.isfinite(v) or v <= 0 for v in (rare_boost, foreground_boost)):
        raise ValueError("Sampler weights must be finite and positive")
    weights = []
    for name in dataset.sample_list:
        _, label, _ = dataset._read(dataset.roots[0] / f"{name}.npz")
        if label is None:
            raise ValueError(f"Missing target label for {name}")
        weights.append(rare_boost if any((label == c).any() for c in rare_classes)
                       else foreground_boost if (label > 0).any() else 1.0)
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights),
                                 replacement=True, generator=generator)
