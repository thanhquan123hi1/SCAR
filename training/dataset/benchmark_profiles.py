"""Named benchmark identities and a versioned complete-modality patient cohort."""
from __future__ import annotations

import json
from pathlib import Path

DATASET_IDS = ("myops380", "myopspp_bc80")
ROI_PROFILES = {
    "myopspp_roi128_76": {"export_id": "roi128mm_76cases", "crop_mm": 128, "spacing_xy": 1.0,
                         "excluded": ("Case2013", "Case2017", "Case2018", "Case2031")},
    "myopspp_roi160_80": {"export_id": "roi160mm_to128_80cases", "crop_mm": 160, "spacing_xy": 1.25,
                         "excluded": ()},
}
DATASET_IDS += tuple(ROI_PROFILES)
MYOPSPP_IDS = {f"Case{i}" for i in range(2001, 2036)} | {f"Case{i}" for i in range(3001, 3046)}
MYOPSPP_SPLIT_PATH = Path(__file__).resolve().parents[2] / "preprocessing/splits/myopspp_bc80/patients.json"


def fixed_myopspp_splits():
    """Return copies; model random seeds never regenerate benchmark patient splits."""
    record = json.loads(MYOPSPP_SPLIT_PATH.read_text(encoding="utf-8"))
    splits = record["patients"]
    all_ids = [name for ids in splits.values() for name in ids]
    if record["dataset_id"] != "myopspp_bc80" or len(all_ids) != 80 or set(all_ids) != MYOPSPP_IDS:
        raise ValueError("Invalid versioned MyoPS++ cohort manifest")
    for split, b, c in (("train", 22, 29), ("val", 6, 7), ("test_vol", 7, 9)):
        if sum(n.startswith("Case2") for n in splits[split]) != b or sum(n.startswith("Case3") for n in splits[split]) != c:
            raise ValueError("Invalid center stratification in MyoPS++ manifest")
    return {name: list(ids) for name, ids in splits.items()}


def fixed_roi_splits(dataset_id):
    excluded = set(ROI_PROFILES[dataset_id]["excluded"])
    return {split: [case for case in cases if case not in excluded]
            for split, cases in fixed_myopspp_splits().items()}
