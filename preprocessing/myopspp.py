"""MyoPS++ raw-label semantics shared by export adapters."""
from __future__ import annotations

import numpy as np

# LV/RV are background for the four-class myocardial pathology task.
RAW_MAPPING = {0: 0, 200: 1, 500: 0, 600: 0, 1220: 2, 2221: 3}


def canonicalize_myopspp(label):
    label = np.asarray(label)
    if not label.size or not np.isfinite(label).all() or not np.isin(label, list(RAW_MAPPING)).all():
        raise ValueError("MyoPS++ masks require integer raw labels 0/200/500/600/1220/2221")
    result = np.zeros(label.shape, dtype=np.uint8)
    for raw, canonical in RAW_MAPPING.items():
        result[label == raw] = canonical
    return result
