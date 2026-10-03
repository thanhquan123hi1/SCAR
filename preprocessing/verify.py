"""Validate MyoPS labels, synchronized modalities, splits and declared geometry."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocessing.preprocessing import MODALITIES
from training.dataset.data_contract import read_split_names, validate_patient_splits, patient_id, lock_benchmark_data
from training.metrics.surface_distance import protocol_for_dataset
from training.dataset.myops_dataset import MyopsDataset


def verify_dataset(data_root, list_dir, label_order="legacy", dataset_id="myops380"):
    data_lock = lock_benchmark_data(data_root, list_dir, label_order, dataset_id=dataset_id)
    splits = {s: read_split_names(list_dir, s) for s in ("train", "val", "test_vol")}
    patients = validate_patient_splits(splits)
    if not all(splits.values()):
        raise ValueError("All train/val/test splits must be nonempty")
    volumes = read_split_names(list_dir, "val_vol")
    if set(volumes) != {patient_id(n) for n in splits["val"]}:
        raise ValueError("Validation volume IDs differ from validation patients")
    counts, with_mm, total = {}, 0, 0
    for split in ("train", "val", "val_vol", "test_vol"):
        folder = split + "_h5" if split.endswith("vol") else "train_npz"
        roots = [Path(data_root) / m / folder for m in MODALITIES]
        dataset = MyopsDataset(*roots, list_dir, split, label_order=label_order)
        for sample in dataset:
            if dataset_id == "myopspp_bc80" and not sample["has_geometry"]:
                raise ValueError("MyoPS++ benchmark requires mm geometry for every sample")
            with_mm += int(sample["has_geometry"])
            total += 1
        counts[split] = len(dataset)
    result = {"status": "passed", "sample_counts": counts,
              "benchmark_protocol": protocol_for_dataset(dataset_id), "benchmark_data": data_lock,
              "patient_counts": {k: len(v) for k, v in patients.items()},
              "samples_with_mm_geometry": with_mm,
              "samples_without_mm_geometry": total - with_mm,
              "geometry_note": ("MyoPS380 benchmark uses voxel HD95 on the unit grid; physical spacing is not required or inferred."
                                if dataset_id == "myops380" else "MyoPS++ benchmark requires native orthogonal mm geometry.")}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="E:/STUDY/DATASET/MyoPS380/Processed_data")
    parser.add_argument("--list-dir", default=str(ROOT / "data/processed/splits"))
    parser.add_argument("--label-order", choices=("legacy", "canonical"), default="legacy")
    parser.add_argument("--dataset", dest="dataset_id", choices=("myops380", "myopspp_bc80"), default="myops380")
    result = verify_dataset(**vars(parser.parse_args(argv)))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
