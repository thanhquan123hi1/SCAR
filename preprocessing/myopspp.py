"""Package the complete C0/LGE/T2 MyoPS++ B35/C45 cohort as an independent benchmark.

Preserves native grids. Header agreement does not establish anatomical alignment;
use aligned data or perform registration upstream. Never chooses an ROI using GT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import nibabel as nib
import numpy as np
from tqdm import tqdm

from preprocessing.preprocessing import MODALITIES, load_aligned_images
from preprocessing.process_and_save import _write_volume, source_sha256
from training.dataset.benchmark_profiles import MYOPSPP_IDS, fixed_myopspp_splits
from training.dataset.data_contract import CANONICAL_LABEL_ORDER, CLASS_NAMES, _write_manifest

RAW_MAPPING = {0: 0, 200: 1, 500: 0, 600: 0, 1220: 2, 2221: 3}


def canonicalize_myopspp(label):
    label = np.asarray(label)
    if not label.size or not np.isfinite(label).all() or not np.isin(label, list(RAW_MAPPING)).all():
        raise ValueError("MyoPS++ masks require integer raw labels 0/200/500/600/1220/2221")
    result = np.zeros(label.shape, dtype=np.uint8)
    for raw, canonical in RAW_MAPPING.items():
        result[label == raw] = canonical
    return result


def discover_myopspp_cases(src_path):
    root = Path(src_path).resolve()
    if (root / "MyoPS_train").is_dir():
        root /= "MyoPS_train"
    cases = {}
    for center in ("CenterB", "CenterC"):
        if not (root / center).is_dir():
            raise FileNotFoundError(f"Missing {center} in {root}")
        for directory in sorted((root / center).iterdir()):
            if not directory.is_dir():
                continue
            case = directory.name
            center_ids = {name for name in MYOPSPP_IDS if name.startswith("Case2" if center == "CenterB" else "Case3")}
            if case not in center_ids:
                raise ValueError(f"{case}: patient ID does not belong to {center} in the fixed cohort")
            if case in cases:
                raise ValueError(f"Duplicate patient ID {case}")
            paths = {}
            for modality, suffix in (("bSSFP", "C0"), ("LGE", "LGE"), ("T2w", "T2"), ("label", "gd")):
                matches = [directory / f"{case}_{suffix}{ext}" for ext in (".nii", ".nii.gz")]
                matches = [p for p in matches if p.is_file()]
                if len(matches) != 1:
                    raise ValueError(f"{case}: expected exactly one {suffix} NIfTI, found {len(matches)}")
                paths[modality] = matches[0]
            cases[case] = paths
    if set(cases) != MYOPSPP_IDS:
        raise ValueError(f"myopspp_bc80 requires B35/C45 cohort; missing={sorted(MYOPSPP_IDS-set(cases))}, extra={sorted(set(cases)-MYOPSPP_IDS)}")
    return cases


def load_myopspp_case(paths):
    images, spacing, affine, unit = load_aligned_images({m: paths[m] for m in MODALITIES}, "percentile")
    reference = nib.load(str(paths["bSSFP"]))
    target = nib.load(str(paths["label"]))
    if target.shape != reference.shape or not np.allclose(target.affine, affine, atol=1e-4, rtol=1e-4):
        raise ValueError("Target has incompatible shape/affine; registration required upstream")
    if unit != "mm" or target.header.get_xyzt_units()[0] != "mm":
        raise ValueError("myopspp_bc80 requires NIfTI geometry declared in mm")
    directions = affine[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4, rtol=1e-4):
        raise ValueError("Physical distance metrics require an orthogonal voxel grid; resample upstream")
    return images, canonicalize_myopspp(target.get_fdata(dtype=np.float32)), spacing, affine


def preprocess_myopspp(src_path, dst_path, show_progress=True):
    """Publish a new complete cache atomically; refuse all existing destinations."""
    source, destination = Path(src_path).resolve(), Path(dst_path).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Cache must be outside the raw source tree")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite dataset directory: {destination}")
    cases = discover_myopspp_cases(source)
    patients = fixed_myopspp_splits()
    assignment = {case: split for split, names in patients.items() for case in names}
    lists = {"train": [], "val": [], "val_vol": patients["val"], "test_vol": patients["test_vol"]}
    info = {"schema_version": 1, "dataset_id": "myopspp_bc80", "class_names": list(CLASS_NAMES),
            "label_order": CANONICAL_LABEL_ORDER, "source_label_encoding": "myopspp_raw_0_200_500_600_1220_2221",
            "raw_mapping": RAW_MAPPING, "normalization": "percentile", "normalization_percentiles": [1, 99],
            "spacing_unit": "mm", "axis_order": "HWD", "source_path": str(source),
            "split_seed": 1234, "patients": patients, "case_shapes": {}, "centers": {},
            "source_checksums_sha256": {}, "pathology_voxels": {},
            "registration": "Matching shape/affine validated; anatomical registration required upstream. No GT crop."}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".building-", dir=destination.parent))
    try:
        for case, paths in tqdm(cases.items(), desc="Package MyoPS++ B/C", disable=not show_progress):
            try:
                images, label, spacing, affine = load_myopspp_case(paths)
            except (ValueError, OSError) as exc:
                raise ValueError(f"{case}: {exc}") from exc
            info["case_shapes"][case] = list(label.shape)
            info["centers"][case] = paths["bSSFP"].parent.parent.name
            info["pathology_voxels"][case] = {"edema": int((label == 2).sum()), "scar": int((label == 3).sum())}
            info["source_checksums_sha256"][case] = {m: source_sha256(p) for m, p in paths.items()}
            metadata = dict(label_order=CANONICAL_LABEL_ORDER, spacing=spacing, affine=affine,
                            source_spacing=spacing, source_affine=affine, spacing_unit="mm",
                            source_spatial_unit="mm", patient_id=case, normalization="percentile")
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
        info["sample_counts"] = {split: len(names) for split, names in lists.items()}
        (staging / "dataset_metadata.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        staging.rename(destination)
    except BaseException:
        # Only remove the staging directory created by this invocation, under the checked parent.
        if staging.exists() and staging.resolve().parent == destination.parent and staging.name.startswith(destination.name + ".building-"):
            shutil.rmtree(staging)
        raise
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-path", default="E:/STUDY/DATASET/Myo_train")
    parser.add_argument("--dst-path", default="data/myopspp_bc80/cache")
    args = parser.parse_args(argv)
    info = preprocess_myopspp(args.src_path, args.dst_path)
    print(json.dumps({"dataset_id": info["dataset_id"], "sample_counts": info["sample_counts"]}, indent=2))
    return info


if __name__ == "__main__":
    main()
