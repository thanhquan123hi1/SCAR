"""Adapt already exported oracle cardiac ROIs to SCAR caches, without reprocessing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import nibabel as nib
import numpy as np
from tqdm import tqdm

from preprocessing.myopspp import RAW_MAPPING, canonicalize_myopspp
from preprocessing.preprocessing import MODALITIES
from preprocessing.process_and_save import _write_volume, source_sha256
from training.dataset.benchmark_profiles import ROI_PROFILES, fixed_roi_splits
from training.dataset.data_contract import CANONICAL_LABEL_ORDER, CLASS_NAMES, _write_manifest
from training.metrics.surface_distance import protocol_for_dataset

NORMALIZATION = "export_percentile_preserved"


def verified_export_files(source):
    """Verify portable relative checksums, and require every consumed file to be covered."""
    records = {}
    for line in (source / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = (source / relative).resolve()
        if (source not in path.parents or len(digest) != 64 or path in records
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Invalid export checksum manifest")
        if source_sha256(path) != digest:
            raise ValueError(f"Export checksum mismatch: {relative}")
        records[path] = digest
    return records


def load_roi_case(paths, metadata, profile):
    native_spacing = np.asarray(metadata.get("native_spacing_mm"), dtype=float)
    native_affine = np.asarray(metadata.get("native_affine"), dtype=float)
    if (native_spacing.shape != (3,) or not np.isfinite(native_spacing).all() or (native_spacing <= 0).any()
            or native_affine.shape != (4, 4) or not np.isfinite(native_affine).all()
            or not np.allclose(native_affine[3], [0, 0, 0, 1])
            or np.linalg.matrix_rank(native_affine[:3, :3]) != 3
            or not np.allclose(np.linalg.norm(native_affine[:3, :3], axis=0), native_spacing, atol=1e-4, rtol=1e-4)):
        raise ValueError("Invalid native source spacing/affine in ROI provenance")
    reference = nib.load(str(paths["bSSFP"]))
    affine = np.asarray(reference.affine)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if (len(reference.shape) != 3 or reference.shape[:2] != (128, 128)
            or reference.shape[2] < 1 or not np.isfinite(affine).all()
            or not np.isfinite(spacing).all() or (spacing <= 0).any()
            or not np.allclose(spacing[:2], profile["spacing_xy"], atol=1e-5)
            or not np.allclose((affine[:3, :3] / spacing).T @ (affine[:3, :3] / spacing), np.eye(3), atol=1e-4)):
        raise ValueError("ROI requires aligned 128x128xD orthogonal mm grid with profile spacing")
    if (metadata.get("crop_mm") != profile["crop_mm"]
            or metadata.get("output_shape") != list(reference.shape)
            or not np.allclose(metadata.get("output_affine"), affine, atol=1e-4)
            or not np.allclose(metadata.get("output_spacing_mm"), spacing, atol=1e-4)
            or not np.isclose(native_spacing[2], spacing[2])
            or "oracle" not in metadata.get("localization", "").lower()):
        raise ValueError("ROI geometry/provenance differs from the exported case metadata")
    images, label = {}, None
    for modality, path in paths.items():
        volume = nib.load(str(path))
        if (volume.shape != reference.shape or volume.header.get_xyzt_units()[0] != "mm"
                or not np.allclose(volume.affine, affine, atol=1e-4, rtol=1e-4)):
            raise ValueError(f"{modality}: inconsistent ROI shape/affine/mm units")
        data = volume.get_fdata(dtype=np.float32)
        if modality == "label":
            label = canonicalize_myopspp(data)
        else:
            if not np.isfinite(data).all() or data.min() < 0 or data.max() > 1:
                raise ValueError(f"{modality}: expected finite already-normalized [0,1] MRI")
            images[modality] = data  # Deliberately no normalization, resampling or cropping.
    return images, label, spacing, affine


def preprocess_myopspp_roi(src_path, dst_path, dataset_id, show_progress=True):
    source, destination = Path(src_path).resolve(), Path(dst_path).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Cache must be outside the export source tree")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite dataset directory: {destination}")
    if dataset_id not in ROI_PROFILES:
        raise ValueError(f"Unknown ROI profile: {dataset_id}")
    profile = ROI_PROFILES[dataset_id]
    checksums = verified_export_files(source)
    def read_record(path):
        if path.resolve() not in checksums:
            raise ValueError(f"Consumed file is missing from export checksums: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    manifest = read_record(source / "dataset_manifest.json")
    patients = fixed_roi_splits(dataset_id)
    cohort = {c for names in patients.values() for c in names}
    if (manifest.get("dataset_id") != profile["export_id"] or manifest.get("crop_mm") != profile["crop_mm"]
            or manifest.get("model_hw") != [128, 128]
            or manifest.get("output_inplane_spacing_mm") != profile["spacing_xy"]
            or manifest.get("patient_count") != len(cohort) or set(manifest.get("patients", [])) != cohort
            or len(manifest.get("patients", [])) != len(cohort)
            or set(manifest.get("excluded_cases", [])) != set(profile["excluded"])
            or manifest.get("normalization") != "p1/p99 after crop/resize; [0,1]"
            or "oracle" not in manifest.get("localization", "").lower()):
        raise ValueError("Export belongs to a different ROI profile, cohort or normalization")
    if read_record(source / "splits.json").get("patients") != patients:
        raise ValueError("Export patient splits differ from the fixed ROI benchmark")
    directories = list((source / "cases").glob("*/*"))
    if len(directories) != len(cohort) or {p.name for p in directories} != cohort:
        raise ValueError("Incomplete or extra ROI patient directories")
    assignment = {case: split for split, names in patients.items() for case in names}
    lists = {"train": [], "val": [], "val_vol": patients["val"], "test_vol": patients["test_vol"]}
    info = dict(schema_version=1, dataset_id=dataset_id, class_names=list(CLASS_NAMES),
                label_order=CANONICAL_LABEL_ORDER, raw_mapping=RAW_MAPPING,
                source_label_encoding="myopspp_raw_0_200_500_600_1220_2221",
                normalization=NORMALIZATION, normalization_percentiles=[1, 99], spacing_unit="mm", axis_order="HWD",
                source_path=str(source), patients=patients, case_shapes={}, centers={}, case_provenance={},
                source_checksums_sha256={}, pathology_voxels={}, export_manifest=manifest,
                export_checksums_sha256=source_sha256(source / "checksums.sha256"),
                benchmark_protocol=protocol_for_dataset(dataset_id))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".building-", dir=destination.parent))
    try:
        for directory in tqdm(sorted(directories), desc=f"Package {dataset_id}", disable=not show_progress):
            case, center = directory.name, directory.parent.name
            if center != ("CenterB" if case.startswith("Case2") else "CenterC"):
                raise ValueError(f"{case}: incorrect center")
            provenance = read_record(directory / "metadata.json")
            if any(provenance.get(k) != v for k, v in (("case", case), ("center", center), ("split", assignment[case]))):
                raise ValueError(f"{case}: incorrect case identity or split metadata")
            paths = {m: directory / f"{case}_{suffix}.nii.gz"
                     for m, suffix in (("bSSFP", "C0"), ("LGE", "LGE"), ("T2w", "T2"), ("label", "gd"))}
            if any(p.resolve() not in checksums for p in paths.values()):
                raise ValueError(f"{case}: modality or target missing from export checksums")
            images, label, spacing, affine = load_roi_case(paths, provenance, profile)
            info["case_shapes"][case] = list(label.shape)
            info["centers"][case] = center
            info["case_provenance"][case] = provenance
            info["source_checksums_sha256"][case] = {m: checksums[p.resolve()] for m, p in paths.items()}
            info["pathology_voxels"][case] = {"edema": int((label == 2).sum()), "scar": int((label == 3).sum())}
            metadata = dict(label_order=CANONICAL_LABEL_ORDER, spacing=spacing, affine=affine,
                            source_spacing=np.asarray(provenance["native_spacing_mm"]),
                            source_affine=np.asarray(provenance["native_affine"]), spacing_unit="mm",
                            source_spatial_unit="mm", patient_id=case, normalization=NORMALIZATION)
            split = assignment[case]
            if split in ("train", "val"):
                for depth in range(label.shape[2]):
                    name = f"{case}_slice{depth:03d}"
                    lists[split].append(name)
                    for modality in MODALITIES:
                        folder = staging / modality / "train_npz"
                        folder.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(folder / f"{name}.npz", image=images[modality][:, :, depth],
                                            label=label[:, :, depth], axis_order="HW", slice_index=depth,
                                            schema_version=1, **metadata)
            if split in ("val", "test_vol"):
                folder = "val_vol_h5" if split == "val" else "test_vol_h5"
                for modality in MODALITIES:
                    _write_volume(staging / modality / folder / f"{case}.npy.h5", images[modality], label, metadata)
        for split, names in lists.items():
            _write_manifest(staging / "lists" / f"{split}.txt", names)
        info["sample_counts"] = {s: len(names) for s, names in lists.items()}
        (staging / "dataset_metadata.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        staging.rename(destination)
    except BaseException:
        if staging.exists() and staging.resolve().parent == destination.parent and staging.name.startswith(destination.name + ".building-"):
            shutil.rmtree(staging)
        raise
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(ROI_PROFILES))
    parser.add_argument("--src-path", required=True, help="Extracted ROI export containing dataset_manifest.json")
    parser.add_argument("--dst-path", required=True, help="New cache directory; existing directories are refused")
    args = parser.parse_args(argv)
    info = preprocess_myopspp_roi(args.src_path, args.dst_path, args.dataset)
    print(json.dumps({"dataset_id": info["dataset_id"], "sample_counts": info["sample_counts"]}, indent=2))
    return info


if __name__ == "__main__":
    main()
