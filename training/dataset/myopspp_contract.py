"""Data locks for the predicted-ROI 64/16 cohort, independent of MyoPS380."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from training.dataset.benchmark_profiles import MYOPSPP_DATASET, MYOPSPP_LOCALIZATION, fixed_myopspp_splits
from training.dataset.data_contract import (
    CANONICAL_LABEL_ORDER, CLASS_NAMES, _write_manifest, patient_id, read_split_names,
    resolve_label_order, validate_patient_splits,
)


def read_myopspp_splits(list_dir):
    root = Path(list_dir)
    if any((root / f"{s}.txt").exists() for s in ("val", "val_vol")):
        raise ValueError("MyoPS++ predicted ROI requires train/test manifests without validation")
    splits = {s: read_split_names(root, s) for s in ("train", "test_vol")}
    patients = validate_patient_splits(splits)
    fixed = fixed_myopspp_splits()
    if any(patients[s] != set(fixed[s]) for s in fixed) or set(splits["test_vol"]) != set(fixed["test_vol"]):
        raise ValueError("MyoPS++ predicted ROI requires the fixed 64 train / 16 test patients")
    indices = {}
    for name in splits["train"]:
        match = re.fullmatch(r"(Case\d{4})_slice(\d{3})", name)
        if match is None:
            raise ValueError(f"Invalid MyoPS++ slice ID: {name}")
        indices.setdefault(match[1], []).append(int(match[2]))
    if any(sorted(values) != list(range(len(values))) for values in indices.values()):
        raise ValueError("MyoPS++ slices must be contiguous from zero")
    return splits


def ensure_myopspp_splits(list_dir, output_dir=None):
    source = read_myopspp_splits(list_dir)
    destination = Path(output_dir) if output_dir is not None else Path(list_dir)
    present = [(destination / f"{s}.txt").exists() for s in source]
    if any(present):
        if not all(present) or read_myopspp_splits(destination) != source:
            raise ValueError("Saved MyoPS++ manifests differ from the requested source")
    else:
        if any((destination / f"{s}.txt").exists() for s in ("val", "val_vol")):
            raise ValueError("MyoPS++ predicted ROI requires no validation manifests")
        for split, names in source.items():
            _write_manifest(destination / f"{split}.txt", names)
    return destination


def lock_myopspp_data(data_root, list_dir, label_order):
    from training.metrics.surface_distance import protocol_for_dataset
    if label_order == "auto" or resolve_label_order(label_order) != CANONICAL_LABEL_ORDER:
        raise ValueError("MyoPS++ predicted ROI requires explicit canonical labels")
    root = Path(data_root)
    metadata = root / "dataset_metadata.json"
    info = json.loads(metadata.read_text(encoding="utf-8"))
    fixed = fixed_myopspp_splits()
    cohort = set(sum(fixed.values(), []))
    mapping = {"0": 0, "200": 1, "500": 0, "600": 0, "1220": 2, "2221": 3}
    if (info.get("dataset_id") != MYOPSPP_DATASET or info.get("label_order") != CANONICAL_LABEL_ORDER
            or info.get("class_names") != list(CLASS_NAMES) or info.get("raw_mapping") != mapping):
        raise ValueError("Cache dataset identity or canonical label mapping differs from MyoPS++ predicted ROI")
    if (info.get("normalization") != "export_percentile_preserved" or info.get("spacing_unit") != "mm"
            or info.get("localization") != MYOPSPP_LOCALIZATION or info.get("no_val") is not True
            or info.get("benchmark_protocol") != protocol_for_dataset(MYOPSPP_DATASET)):
        raise ValueError("MyoPS++ predicted ROI preprocessing/evaluation protocol differs")
    shapes = info.get("case_shapes", {})
    if info.get("patients") != fixed or set(shapes) != cohort:
        raise ValueError("Cache cohort/splits differ from the fixed MyoPS++ benchmark")
    if any(len(s) != 3 or s[:2] != [128, 128] or type(s[2]) is not int or s[2] <= 0 for s in shapes.values()):
        raise ValueError("MyoPS++ predicted ROI requires 128x128xD shapes")
    splits = read_myopspp_splits(list_dir)
    expected = {f"{case}_slice{i:03d}" for case in fixed["train"] for i in range(shapes[case][2])}
    if set(splits["train"]) != expected:
        raise ValueError("Incomplete or changed training slices for the fixed MyoPS++ patients")
    counts = {s: len(names) for s, names in splits.items()}
    if info.get("sample_counts") != counts:
        raise ValueError("MyoPS++ sample counts differ from manifests")
    files = [metadata]
    for modality in ("bSSFP", "LGE", "T2w"):
        for split, folder, suffix in (("train", "train_npz", ".npz"), ("test_vol", "test_vol_h5", ".npy.h5")):
            paths = list((root / modality / folder).glob("*" + suffix))
            if {p.name[:-len(suffix)] for p in paths} != set(splits[split]):
                raise ValueError(f"Cache inventory differs from MyoPS++ manifests: {modality}/{folder}")
            files.extend(paths)
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        with path.open("rb") as stream:
            file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        digest.update(f"{path.relative_to(root).as_posix()}\0{file_hash}\n".encode())
    return {"schema_version": 1, "dataset_id": MYOPSPP_DATASET, "source_label_order": CANONICAL_LABEL_ORDER,
            "canonical_class_names": list(CLASS_NAMES), "cache_sha256": digest.hexdigest(),
            "cache_files": len(files), "patient_counts": {s: len(ids) for s, ids in fixed.items()},
            "sample_counts": counts,
            "split_ids_sha256": {s: hashlib.sha256(("\n".join(sorted(names)) + "\n").encode()).hexdigest()
                                 for s, names in splits.items()}}
