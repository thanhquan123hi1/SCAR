"""Supported datasets and the immutable predicted-ROI MyoPS++ patient split."""
from __future__ import annotations

import json
from pathlib import Path

MYOPSPP_DATASET = "myopspp_unet2d_roi128_1mm"
MYOPSPP_EXPORT_ID = "MyoPSpp_UNet2D_ROI128_1mm_2026-10-27"
MYOPSPP_LOCALIZATION = "predicted_myocardium_unet2d_oof_train_final_val_test"
MYOPSPP_SOURCE = "D:/NCKH/Dataset/MyoPSpp_UNet2D_ROI128_1mm"
DATASET_IDS = ("myops380", MYOPSPP_DATASET)
ROI_PROFILES = {
    MYOPSPP_DATASET: {"export_id": MYOPSPP_EXPORT_ID, "crop_mm": 128, "spacing_xy": 1.0},
}
MYOPSPP_IDS = {f"Case{i}" for i in range(2001, 2036)} | {f"Case{i}" for i in range(3001, 3046)}
MYOPSPP_SPLIT_PATH = Path(__file__).resolve().parents[2] / f"preprocessing/splits/{MYOPSPP_DATASET}/patients.json"


def fixed_myopspp_splits():
    """64 train / 16 test, preserved from the export; no seed-dependent validation."""
    record = json.loads(MYOPSPP_SPLIT_PATH.read_text(encoding="utf-8"))
    splits = record["patients"]
    all_ids = [name for ids in splits.values() for name in ids]
    if (record["dataset_id"] != MYOPSPP_EXPORT_ID or set(splits) != {"train", "test_vol"}
            or len(all_ids) != 80 or len(set(all_ids)) != 80 or set(all_ids) != MYOPSPP_IDS):
        raise ValueError("Invalid versioned MyoPS++ predicted-ROI cohort manifest")
    for split, b, c in (("train", 28, 36), ("test_vol", 7, 9)):
        if (sum(n.startswith("Case2") for n in splits[split]) != b
                or sum(n.startswith("Case3") for n in splits[split]) != c):
            raise ValueError("Invalid center stratification in MyoPS++ manifest")
    return {name: list(ids) for name, ids in splits.items()}
