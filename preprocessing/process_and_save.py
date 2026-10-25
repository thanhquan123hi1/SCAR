"""Package already registered MyoPS NIfTI into leakage-free NPZ/H5 datasets.

This is packaging and intensity normalization, not image registration. Shape and
affine agreement are necessary, but cannot prove anatomical registration. Supply
the aligned release or register independently upstream; target masks are never
used to choose a crop or estimate a transformation here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset.data_contract import (
    CANONICAL_LABEL_ORDER,
    CLASS_NAMES,
    _write_manifest,
    read_split_names,
    resolve_label_order,
    validate_patient_splits,
)

from preprocessing.preprocessing import MODALITIES, discover_cases, load_aligned_case, create_patient_splits


def source_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_volume(path, image, label, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as data:
        data.create_dataset("image", data=image, compression="gzip", compression_opts=1)
        data.create_dataset("label", data=label, compression="gzip", compression_opts=1)
        data.create_dataset("spacing", data=metadata["spacing"])
        data.create_dataset("affine", data=metadata["affine"])
        for name in ("label_order", "patient_id", "normalization", "spacing_unit", "source_spatial_unit"):
            data.attrs[name] = metadata[name]
        data.create_dataset("source_affine", data=metadata["source_affine"])
        data.create_dataset("source_spacing", data=metadata["source_spacing"])
        data.attrs["axis_order"] = "HWD"
        data.attrs["schema_version"] = 1


def preprocess_dataset(src_path, dst_path, list_dir=None, test_list=None, seed=42,
                       test_fraction=0.2, val_fraction=0.2, label_order="legacy",
                       normalization="unit255", show_progress=True, spatial_unit=None):
    """Write a new dataset directory; never overwrite an existing dataset."""
    src_path, dst_path = Path(src_path).resolve(), Path(dst_path).resolve()
    list_dir = Path(list_dir).resolve() if list_dir else dst_path / "lists"
    if dst_path == src_path or src_path in dst_path.parents:
        raise ValueError("Destination must not be the raw source or inside it.")
    if dst_path.exists() and any(dst_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty dataset directory: {dst_path}")
    if list_dir.exists() and any(list_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing split manifests: {list_dir}")
    cases = discover_cases(src_path)
    test_ids = None
    if test_list:
        path = Path(test_list)
        test_ids = read_split_names(path.parent, path.stem)
    patient_splits = create_patient_splits(list(cases), seed, test_fraction, val_fraction, test_ids)
    validate_patient_splits(patient_splits)
    assignment = {case: split for split, names in patient_splits.items() for case in names}
    lists = {"train": [], "val": [], "val_vol": patient_splits["val"], "test_vol": patient_splits["test_vol"]}
    dst_path.mkdir(parents=True, exist_ok=True)
    for case in tqdm(cases, desc="Package aligned CMR", disable=not show_progress):
        try:
            images, label, spacing, affine = load_aligned_case(cases[case], label_order, normalization)
        except (ValueError, OSError) as exc:
            raise ValueError(f"Case {case}: {exc}") from exc
        source_unit = nib.load(str(cases[case]["bSSFP"])).header.get_xyzt_units()[0]
        if spatial_unit is not None and source_unit not in ("unknown", spatial_unit):
            raise ValueError(f"Explicit spatial unit conflicts with NIfTI header for {case}.")
        effective_unit = spatial_unit or source_unit
        scale = {"meter": 1000.0, "mm": 1.0, "micron": 0.001}.get(effective_unit)
        output_affine = affine.copy()
        if scale is not None:
            output_affine[:3, :] *= scale
        metadata = {
            "label_order": CANONICAL_LABEL_ORDER,
            "spacing": spacing * scale if scale is not None else spacing,
            "affine": output_affine,
            "source_affine": affine,
            "source_spacing": spacing,
            "spacing_unit": "mm" if scale is not None else "unknown",
            "source_spatial_unit": source_unit,
            "patient_id": case,
            "normalization": normalization,
        }
        split = assignment[case]
        if split in ("train", "val"):
            for depth in range(label.shape[2]):
                slice_name = f"{case}_slice{depth:03d}"
                lists[split].append(slice_name)
                for modality in MODALITIES:
                    directory = dst_path / modality / "train_npz"
                    directory.mkdir(parents=True, exist_ok=True)
                    np.savez(directory / f"{slice_name}.npz", image=images[modality][:, :, depth],
                             label=label[:, :, depth], axis_order="HW", slice_index=depth,
                             schema_version=1, **metadata)
        if split in ("val", "test_vol"):
            folder = "val_vol_h5" if split == "val" else "test_vol_h5"
            for modality in MODALITIES:
                _write_volume(dst_path / modality / folder / f"{case}.npy.h5", images[modality], label, metadata)
    for split, names in lists.items():
        _write_manifest(list_dir / f"{split}.txt", names)
    info = {
        "schema_version": 1,
        "class_names": CLASS_NAMES,
        "label_order": CANONICAL_LABEL_ORDER,
        "source_label_order": resolve_label_order(label_order),
        "normalization": normalization,
        "spatial_unit_override": spatial_unit,
        "axis_order": "HWD",
        "source_path": str(src_path),
        "seed": seed,
        "test_fraction": test_fraction if test_ids is None else None,
        "test_manifest": str(Path(test_list).resolve()) if test_list else None,
        "val_fraction_of_non_test_patients": val_fraction,
        "patients": patient_splits,
        "sample_counts": {split: len(names) for split, names in lists.items()},
        "source_checksums_sha256": {case: {modality: source_sha256(path) for modality, path in paths.items()} for case, paths in cases.items()},
        "registration": "Required upstream. Matching shape and affine validated; no registration or mask-derived crop performed.",
    }
    (dst_path / "dataset_metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_path", "--src-path", default="E:/STUDY/DATASET/MyoPS380/Raw_data")
    parser.add_argument("--dst_path", "--dst-path", default=str(PROJECT_ROOT / "data" / "processed" / "cache"))
    parser.add_argument("--list-dir", default=None, help="Defaults to DST/lists; must be empty or absent.")
    parser.add_argument("--test-list", default=str(PROJECT_ROOT / "preprocessing" / "splits" / "test_vol.txt"), help="Preserve an existing held-out patient manifest, e.g. list/test_vol.txt.")
    parser.add_argument("--label-order", choices=("legacy", "canonical"), default="legacy", help="Source NIfTI encoding: legacy is 0 bg, 1 normal, 2 scar, 3 edema. Output is always canonical.")
    parser.add_argument("--normalization", choices=("unit255", "unit", "percentile"), default="unit255")
    parser.add_argument("--spatial-unit", choices=("mm", "meter", "micron"), default=None, help="Explicit physical unit only when header is unknown; known conflicting units are rejected.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of non-test patients held out for validation.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    info = preprocess_dataset(**vars(args))
    print(json.dumps({"sample_counts": info["sample_counts"], "label_order": info["label_order"]}, indent=2))


if __name__ == "__main__":
    main()
