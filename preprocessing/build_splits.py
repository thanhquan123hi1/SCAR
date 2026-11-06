"""Create reproducible patient manifests from an existing three-modality cache."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocessing.preprocessing import MODALITIES, create_patient_splits, discover_cases
from training.dataset.data_contract import (
    _write_manifest, patient_id, read_split_names, validate_patient_splits,
)


def cache_inventory(data_root):
    root = Path(data_root)
    result = {}
    for folder, extension in (("train_npz", ".npz"), ("test_vol_h5", ".npy.h5")):
        inventories = [{p.name[:-len(extension)] for p in (root / m / folder).glob("*" + extension)}
                       for m in MODALITIES]
        if not inventories[0]:
            raise FileNotFoundError(f"No {folder} files found in {root}")
        if any(ids != inventories[0] for ids in inventories[1:]):
            raise ValueError(f"Modality file IDs differ in {folder}")
        for name in inventories[0]:
            patient_id(name)
        result[folder] = inventories[0]
    validate_patient_splits({"non_test": sorted(result["train_npz"]),
                             "test": sorted(result["test_vol_h5"])})
    return result


def build_splits(data_root, list_dir, test_list=None, seed=1234, val_fraction=0.2, raw_root=None, no_val=False):
    inventory = cache_inventory(data_root)
    slices, tests = inventory["train_npz"], inventory["test_vol_h5"]
    cases = {patient_id(name) for name in slices} | tests
    if test_list:
        path = Path(test_list)
        fixed_test = set(read_split_names(path.parent, path.stem))
        if fixed_test != tests:
            raise ValueError("Fixed held-out patient list differs from cached test volumes")
    if raw_root and set(discover_cases(raw_root)) != cases:
        raise ValueError("Raw patient IDs differ from cached patients")
    destination = Path(list_dir)
    if no_val:
        from training.dataset.no_val import validate_no_val_splits, read_no_val_splits
        splits = {"train": sorted(slices), "test_vol": sorted(tests)}
        patients = validate_no_val_splits(splits)
        if any((destination / f"{name}.txt").exists() for name in ("train", "test_vol", "val", "val_vol")):
            if read_no_val_splits(destination) != splits:
                raise ValueError("Existing no-val manifests differ from the complete cache inventory")
        else:
            for name, values in splits.items():
                _write_manifest(destination / f"{name}.txt", values)
        return {"patient_counts": {k: len(v) for k, v in patients.items()},
                "sample_counts": {k: len(v) for k, v in splits.items()},
                "list_dir": str(destination.resolve()), "no_val": True}
    names = ("train", "val", "val_vol", "test_vol")
    present = [(destination / f"{name}.txt").is_file() for name in names]
    if any(present):
        if not all(present):
            raise ValueError("Incomplete manifests; choose a new list-dir instead of mixing old and new splits")
        splits = {name: read_split_names(destination, name) for name in names}
        if set(splits["train"]) | set(splits["val"]) != slices or set(splits["test_vol"]) != tests:
            raise ValueError("Existing manifests do not match the complete cache inventory")
        if set(splits["val_vol"]) != {patient_id(n) for n in splits["val"]}:
            raise ValueError("val_vol IDs must exactly match validation slice patients")
    else:
        assigned = create_patient_splits(sorted(cases), seed=seed, val_fraction=val_fraction,
                                         test_ids=sorted(tests))
        splits = {name: sorted(n for n in slices if patient_id(n) in set(assigned[name]))
                  for name in ("train", "val")}
        splits.update(val_vol=assigned["val"], test_vol=sorted(tests))
    patients = validate_patient_splits({k: splits[k] for k in ("train", "val", "test_vol")})
    if not splits["train"] or not splits["val"]:
        raise ValueError("Training and validation must both be nonempty")
    if not any(present):
        for name, values in splits.items():
            _write_manifest(destination / f"{name}.txt", values)
        metadata = {"seed": seed, "val_fraction_of_non_test_patients": val_fraction,
                    "data_root": str(Path(data_root).resolve()),
                    "fixed_test_list": str(Path(test_list).resolve()) if test_list else None,
                    "patient_counts": {k: len(v) for k, v in patients.items()}}
        (destination / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"patient_counts": {k: len(v) for k, v in patients.items()},
            "sample_counts": {k: len(v) for k, v in splits.items()}, "list_dir": str(destination.resolve())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="E:/STUDY/DATASET/MyoPS380/Processed_data")
    parser.add_argument("--list-dir", default=str(ROOT / "data/processed/splits"))
    parser.add_argument("--test-list", default=str(ROOT / "preprocessing/splits/test_vol.txt"))
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--no-val", action="store_true")
    result = build_splits(**vars(parser.parse_args(argv)))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
