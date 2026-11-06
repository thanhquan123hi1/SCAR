"""Immutable MyoPS380 304/76 cohort; never infer a split from a training seed."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

from training.dataset.data_contract import (
    CLASS_NAMES, OFFICIAL_TEST_IDS_SHA256, _write_manifest, patient_id,
    read_split_names, resolve_label_order, validate_patient_splits,
)

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "preprocessing/splits/myops380_no_val"
TRAIN_IDS_SHA256 = "16248ec3630bb0493fb421aab5ffb4dc9b1d0fd49c8614d69d3098a92305f919"


def ids_hash(names):
    return hashlib.sha256(("\n".join(sorted(names)) + "\n").encode()).hexdigest()


def validate_no_val_splits(splits):
    patients = validate_patient_splits(splits)
    if ids_hash(splits["test_vol"]) != OFFICIAL_TEST_IDS_SHA256:
        raise ValueError("Test patients differ from the official MyoPS380 76-case cohort")
    if (len(patients["train"]) != 304 or len(patients["test_vol"]) != 76
            or patients["train"] | patients["test_vol"] != {f"case{i:04d}" for i in range(1, 381)}):
        raise ValueError("--no-val requires exactly 304 train / 76 test patients")
    reference = read_split_names(REFERENCE_DIR, "train")
    if ids_hash(reference) != TRAIN_IDS_SHA256:
        raise ValueError("Versioned reference training manifest was modified")
    if len(reference) != 1972 or set(splits["train"]) != set(reference):
        raise ValueError("--no-val requires all 1972 official training slices; use myops380_no_val manifests")
    by_patient = {}
    for name in splits["train"]:
        match = re.fullmatch(r"case\d{4}_slice(\d{3})", name)
        if not match:
            raise ValueError(f"Invalid training slice ID: {name}")
        by_patient.setdefault(patient_id(name), []).append(int(match[1]))
    if any(sorted(indices) != list(range(len(indices))) for indices in by_patient.values()):
        raise ValueError("Training slices must be complete and contiguous from zero")
    return patients


def read_no_val_splits(list_dir):
    if any((Path(list_dir) / f"{name}.txt").exists() for name in ("val", "val_vol")):
        raise ValueError("--no-val requires a separate train/test list-dir without val manifests")
    splits = {name: read_split_names(list_dir, name) for name in ("train", "test_vol")}
    validate_no_val_splits(splits)
    return splits


def ensure_no_val_splits(list_dir, output_dir=None):
    source = read_no_val_splits(list_dir)
    destination = Path(output_dir) if output_dir is not None else Path(list_dir)
    present = [(destination / f"{name}.txt").exists() for name in source]
    if any(present):
        if not all(present) or read_no_val_splits(destination) != source:
            raise ValueError("Saved no-val manifests differ from the requested source; do not modify a run")
    else:
        for name, values in source.items():
            _write_manifest(destination / f"{name}.txt", values)
    return destination


def lock_no_val_data(data_root, list_dir, label_order):
    if label_order == "auto":
        raise ValueError("Locked benchmark requires explicit legacy or canonical label order")
    splits = read_no_val_splits(list_dir)
    patients = validate_no_val_splits(splits)
    root, files = Path(data_root), []
    for modality in ("bSSFP", "LGE", "T2w"):
        for split, folder, suffix in (("train", "train_npz", ".npz"),
                                      ("test_vol", "test_vol_h5", ".npy.h5")):
            paths = list((root / modality / folder).glob("*" + suffix))
            if {p.name[:-len(suffix)] for p in paths} != set(splits[split]):
                raise ValueError(f"Cache inventory differs from locked manifests: {modality}/{folder}")
            files.extend(paths)
    metadata = root / "dataset_metadata.json"
    if metadata.is_file():
        files.append(metadata)
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        with path.open("rb") as stream:
            file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        digest.update(f"{path.relative_to(root).as_posix()}\0{file_hash}\n".encode())
    return {"schema_version": 2, "split_protocol": "myops380_train304_test76_last_v1",
            "source_label_order": resolve_label_order(label_order),
            "canonical_class_names": list(CLASS_NAMES), "cache_sha256": digest.hexdigest(),
            "cache_files": len(files), "official_test_ids_sha256": ids_hash(splits["test_vol"]),
            "split_ids_sha256": {s: ids_hash(names) for s, names in splits.items()},
            "patient_counts": {s: len(ids) for s, ids in patients.items()}}
