"""Shared label semantics and reproducible, patient-disjoint split manifests."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re

import numpy as np

CLASS_NAMES = ("background", "normal_myocardium", "edema", "scar")
CANONICAL_LABEL_ORDER = "background_normal_edema_scar"
LEGACY_LABEL_ORDER = "background_normal_scar_edema"
LABEL_ORDERS = {
    "canonical": CANONICAL_LABEL_ORDER,
    "legacy": LEGACY_LABEL_ORDER,
    CANONICAL_LABEL_ORDER: CANONICAL_LABEL_ORDER,
    LEGACY_LABEL_ORDER: LEGACY_LABEL_ORDER,
}

# SHA256 of sorted, newline-terminated official 76 patient IDs (independent of CRLF).
OFFICIAL_TEST_IDS_SHA256 = "3237b31411833cedcb3ecd86d36edae356bd3f880d3fcfd915066f1c850d9cbc"


def lock_benchmark_data(data_root, list_dir, label_order, dataset_id="myops380"):
    """Validate MyoPS380 cohort and fingerprint cache bytes without changing data.

    Persist the returned record in the run/checkpoint. Resume/evaluation compare
    this record, so changing files in place cannot silently change a benchmark.
    """
    from training.dataset.benchmark_profiles import ROI_PROFILES
    if dataset_id == "myopspp_bc80" or dataset_id in ROI_PROFILES:
        return _lock_myopspp_data(data_root, list_dir, label_order, dataset_id)
    if dataset_id != "myops380":
        raise ValueError(f"Unknown benchmark dataset: {dataset_id}")
    if label_order == "auto":
        raise ValueError("Locked benchmark requires explicit legacy or canonical label order")
    order = resolve_label_order(label_order)
    splits = {s: read_split_names(list_dir, s) for s in ("train", "val", "test_vol")}
    patients = validate_patient_splits(splits)
    fixed_hash = hashlib.sha256(("\n".join(sorted(splits["test_vol"])) + "\n").encode()).hexdigest()
    if fixed_hash != OFFICIAL_TEST_IDS_SHA256:
        raise ValueError("Test patients differ from the official MyoPS380 76-case cohort")
    if not patients["train"] or not patients["val"] or len(patients["train"] | patients["val"]) != 304:
        raise ValueError("MyoPS380 requires 304 non-test patients split into nonempty train/val sets")
    if set().union(*patients.values()) != {f"case{i:04d}" for i in range(1, 381)}:
        raise ValueError("MyoPS380 cohort must contain exactly case0001 through case0380")
    if _validate_volume_validation(list_dir, splits["val"]) is None:
        raise ValueError("Locked patient-level selection requires val_vol.txt")
    root = Path(data_root)
    files = []
    for modality in ("bSSFP", "LGE", "T2w"):
        for folder, suffix, expected in (
            ("train_npz", ".npz", set(splits["train"] + splits["val"])),
            ("test_vol_h5", ".npy.h5", set(splits["test_vol"])),
        ):
            paths = list((root / modality / folder).glob("*" + suffix))
            if {p.name[:-len(suffix)] for p in paths} != expected:
                raise ValueError(f"Cache inventory differs from locked manifests: {modality}/{folder}")
            files.extend(paths)
        volumes = list((root / modality / "val_vol_h5").glob("*.npy.h5"))
        if volumes and {p.name[:-7] for p in volumes} != patients["val"]:
            raise ValueError(f"Incomplete validation volume inventory: {modality}")
        files.extend(volumes)
    metadata = root / "dataset_metadata.json"
    if metadata.is_file():
        files.append(metadata)
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        with path.open("rb") as stream:
            file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        digest.update(f"{path.relative_to(root).as_posix()}\0{file_hash}\n".encode())
    return {"schema_version": 1, "source_label_order": order,
            "canonical_class_names": list(CLASS_NAMES), "cache_sha256": digest.hexdigest(),
            "cache_files": len(files), "official_test_ids_sha256": fixed_hash,
            "patient_counts": {s: len(ids) for s, ids in patients.items()}}


def _lock_myopspp_data(data_root, list_dir, label_order, dataset_id="myopspp_bc80"):
    from training.dataset.benchmark_profiles import fixed_myopspp_splits, MYOPSPP_IDS, ROI_PROFILES, fixed_roi_splits

    if label_order == "auto" or resolve_label_order(label_order) != CANONICAL_LABEL_ORDER:
        raise ValueError(f"{dataset_id} requires explicit canonical labels")
    root = Path(data_root)
    metadata = root / "dataset_metadata.json"
    info = json.loads(metadata.read_text(encoding="utf-8"))
    if info.get("dataset_id") != dataset_id or info.get("label_order") != CANONICAL_LABEL_ORDER:
        raise ValueError("Cache belongs to a different dataset/profile or label convention")
    roi = ROI_PROFILES.get(dataset_id)
    normalization = "export_percentile_preserved" if roi else "percentile"
    if info.get("normalization") != normalization or info.get("spacing_unit") != "mm":
        raise ValueError(f"{dataset_id} requires {normalization} normalization and mm geometry")
    fixed = fixed_roi_splits(dataset_id) if roi else fixed_myopspp_splits()
    cohort = set(c for cases in fixed.values() for c in cases) if roi else MYOPSPP_IDS
    if info.get("patients") != fixed or set(info.get("case_shapes", {})) != cohort:
        raise ValueError("Cache cohort/splits differ from the fixed MyoPS++ benchmark")
    if roi:
        from training.metrics.surface_distance import protocol_for_dataset
        if info.get("benchmark_protocol") != protocol_for_dataset(dataset_id):
            raise ValueError("ROI cache preprocessing/evaluation protocol differs from the selected profile")
        for case, shape in info["case_shapes"].items():
            if len(shape) != 3 or shape[:2] != [128, 128] or type(shape[2]) is not int or shape[2] <= 0:
                raise ValueError(f"{case}: ROI requires 128x128xD")
    splits = {s: read_split_names(list_dir, s) for s in ("train", "val", "val_vol", "test_vol")}
    patients = validate_patient_splits({s: splits[s] for s in ("train", "val", "test_vol")})
    for split in ("train", "val"):
        expected = set()
        for case in fixed[split]:
            shape = info["case_shapes"][case]
            if len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
                raise ValueError(f"Invalid native shape for {case}")
            expected.update(f"{case}_slice{i:03d}" for i in range(shape[2]))
        if set(splits[split]) != expected:
            raise ValueError(f"Incomplete or changed {split} slices for the fixed MyoPS++ patients")
    if set(splits["val_vol"]) != set(fixed["val"]) or set(splits["test_vol"]) != set(fixed["test_vol"]):
        raise ValueError("Validation/test patients differ from the fixed MyoPS++ manifests")
    files = [metadata]
    for modality in ("bSSFP", "LGE", "T2w"):
        for folder, suffix, expected in (
            ("train_npz", ".npz", set(splits["train"] + splits["val"])),
            ("val_vol_h5", ".npy.h5", set(fixed["val"])),
            ("test_vol_h5", ".npy.h5", set(fixed["test_vol"])),
        ):
            paths = list((root / modality / folder).glob("*" + suffix))
            if {p.name[:-len(suffix)] for p in paths} != expected:
                raise ValueError(f"Cache inventory differs from MyoPS++ manifests: {modality}/{folder}")
            files.extend(paths)
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        with path.open("rb") as stream:
            file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        digest.update(f"{path.relative_to(root).as_posix()}\0{file_hash}\n".encode())
    return {"schema_version": 1, "dataset_id": dataset_id, "source_label_order": CANONICAL_LABEL_ORDER,
            "canonical_class_names": list(CLASS_NAMES), "cache_sha256": digest.hexdigest(),
            "cache_files": len(files), "patient_counts": {s: len(ids) for s, ids in patients.items()},
            "split_ids_sha256": {s: hashlib.sha256(("\n".join(sorted(names)) + "\n").encode()).hexdigest()
                                 for s, names in splits.items()}}


def resolve_label_order(value):
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return LABEL_ORDERS[str(value)]
    except KeyError as exc:
        raise ValueError(f"Unknown label order {value!r}; use canonical or legacy.") from exc


def canonicalize_label(label, label_order):
    """Convert 0..3 class IDs once, never mutate the source array."""
    label = np.asarray(label)
    if not np.isfinite(label).all() or not np.equal(label, np.round(label)).all():
        raise ValueError("Segmentation masks must contain finite integer class IDs.")
    if label.size == 0 or label.min() < 0 or label.max() > 3:
        raise ValueError(f"Expected class IDs 0..3; found {np.unique(label).tolist()}.")
    result = label.astype(np.uint8, copy=True)
    if resolve_label_order(label_order) == LEGACY_LABEL_ORDER:
        result = np.asarray([0, 1, 3, 2], dtype=np.uint8)[result]
    return result


def patient_id(sample_name):
    """Released layout uses caseXXXX_sliceNNN or caseXXXX volume IDs."""
    name = str(sample_name).strip()
    for extension in (".npy.h5", ".nii.gz", ".npz", ".h5", ".nii"):
        if name.endswith(extension):
            name = name[:-len(extension)]
            break
    if not name or name in {".", ".."} or any(c in name for c in ("/", "\\", ":")):
        raise ValueError(f"Invalid sample name {sample_name!r}; manifests must contain basenames.")
    return re.sub(r"_slice\d+$", "", name)


def read_split_names(list_dir, split):
    path = Path(list_dir) / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate samples in {path}; regenerate manifests without append mode.")
    for name in names:
        patient_id(name)
        if Path(name).suffix:
            raise ValueError(f"Split manifests use IDs without file extensions: {name!r} in {path}")
    return names


def validate_patient_splits(splits):
    """Reject sample duplication and all cross-split patient overlap."""
    patient_sets = {}
    for split, names in splits.items():
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate samples in {split} split.")
        patient_sets[split] = {patient_id(name) for name in names}
    keys = list(patient_sets)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            overlap = patient_sets[left] & patient_sets[right]
            if overlap:
                raise ValueError(f"Patient leakage between {left} and {right}: {sorted(overlap)[:10]}")
    return patient_sets


def _write_manifest(path, names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    temporary.replace(path)


def _validate_volume_validation(list_dir, validation_names):
    if not (Path(list_dir) / "val_vol.txt").is_file():
        return None
    volumes = read_split_names(list_dir, "val_vol")
    if set(volumes) != {patient_id(name) for name in validation_names}:
        raise ValueError("val_vol patient IDs must exactly match the held-out val slice patients.")
    return volumes


def ensure_patient_splits(list_dir, val_fraction=0.2, seed=42, output_dir=None):
    """Return manifests with train/val NPZ IDs and unchanged test IDs.

    Hold out whole training patients only when val.txt does not already exist.
    A complete output directory is validated and reused, making resume stable.
    Use an experiment output_dir to keep original dataset manifests untouched.
    """
    source = Path(list_dir)
    destination = Path(output_dir) if output_dir is not None else source
    if all((destination / f"{s}.txt").is_file() for s in ("train", "val", "test_vol")):
        splits = {s: read_split_names(destination, s) for s in ("train", "val", "test_vol")}
        validate_patient_splits(splits)
        if _validate_volume_validation(destination, splits["val"]) is None:
            _write_manifest(destination / "val_vol.txt", sorted({patient_id(n) for n in splits["val"]}))
        if not splits["train"] or not splits["val"]:
            raise ValueError("Both training and validation splits must be nonempty.")
        return destination
    train = read_split_names(source, "train")
    test = read_split_names(source, "test_vol") if (source / "test_vol.txt").exists() else []
    if (source / "val.txt").is_file():
        val = read_split_names(source, "val")
    else:
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be strictly between 0 and 1.")
        patients = sorted({patient_id(name) for name in train})
        if len(patients) < 2:
            raise ValueError("At least two training patients are required for held-out validation.")
        shuffled = np.random.default_rng(seed).permutation(patients)
        count = min(len(patients) - 1, max(1, round(len(patients) * val_fraction)))
        val_patients = set(shuffled[:count])
        val = [name for name in train if patient_id(name) in val_patients]
        train = [name for name in train if patient_id(name) not in val_patients]
    splits = {"train": train, "val": val, "test_vol": test}
    patients = validate_patient_splits(splits)
    val_volumes = _validate_volume_validation(source, val)
    if not train or not val:
        raise ValueError("Both training and validation splits must be nonempty.")
    for split, names in splits.items():
        _write_manifest(destination / f"{split}.txt", names)
    _write_manifest(destination / "val_vol.txt", val_volumes if val_volumes is not None else sorted(patients["val"]))
    metadata = {
        "seed": seed,
        "val_fraction_of_training_patients": val_fraction,
        "source_list_dir": str(source.resolve()),
        "patient_ids": {split: sorted(ids) for split, ids in patients.items()},
        "sample_counts": {split: len(names) for split, names in splits.items()},
    }
    (destination / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return destination
