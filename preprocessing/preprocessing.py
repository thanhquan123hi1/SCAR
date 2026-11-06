"""Aligned MyoPS discovery, normalization and patient split primitives.

These functions preserve the supplied grid. Registration is required upstream;
matching headers alone cannot establish anatomical registration.
"""
from pathlib import Path

import nibabel as nib
import numpy as np

from training.dataset.data_contract import canonicalize_label

MODALITIES = ("bSSFP", "LGE", "T2w")

def _nifti_index(directory, labels=False):
    directory = Path(directory)
    paths = sorted((*directory.glob("*.nii"), *directory.glob("*.nii.gz")))
    result = {}
    for path in paths:
        name = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
        if labels and name.endswith("_gt"):
            name = name[:-3]
        if name in result:
            raise ValueError(f"Duplicate case ID {name!r} in {directory}.")
        result[name] = path
    if not result:
        raise FileNotFoundError(f"No NIfTI files in {directory}.")
    return result


def discover_cases(src_path):
    """Join by patient ID, never by sorted position in independent file lists."""
    src_path = Path(src_path)
    indices = {modality: _nifti_index(src_path / modality) for modality in MODALITIES}
    indices["label"] = _nifti_index(src_path / "label", labels=True)
    expected = set(indices["bSSFP"])
    for modality, index in indices.items():
        if set(index) != expected:
            raise ValueError(
                f"Patient IDs differ for {modality}: missing={sorted(expected - set(index))[:10]}, extra={sorted(set(index) - expected)[:10]}"
            )
    return {name: {modality: index[name] for modality, index in indices.items()} for name in sorted(expected)}


def normalize_image(image, method="unit255"):
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0 or not np.isfinite(image).all():
        raise ValueError("MRI intensities must be finite and nonempty.")
    if method == "unit255":
        if image.min() < -1e-4 or image.max() > 255 + 1e-4:
            raise ValueError("unit255 normalization requires release intensities in [0,255]. Use --normalization percentile for other intensity scales.")
        return np.clip(image / 255.0, 0, 1)
    if method == "unit":
        if image.min() < -1e-5 or image.max() > 1 + 1e-5:
            raise ValueError("unit normalization requires images already in [0,1].")
        return np.clip(image, 0, 1)
    if method == "percentile":
        foreground = image[image != 0]
        if not foreground.size:
            return np.zeros_like(image)
        low, high = np.percentile(foreground, [1, 99])
        if high <= low:
            return np.zeros_like(image)
        return np.clip((image - low) / (high - low), 0, 1).astype(np.float32)
    raise ValueError(f"Unknown normalization method: {method}")


def load_aligned_case(paths, label_order="legacy", normalization="unit255"):
    volumes = {name: nib.load(str(path)) for name, path in paths.items()}
    reference = volumes["bSSFP"]
    if len(reference.shape) != 3 or any(size == 0 for size in reference.shape):
        raise ValueError(f"Expected a nonempty aligned 3D volume: {paths['bSSFP']} has {reference.shape}.")
    for name, volume in volumes.items():
        if volume.shape != reference.shape or not np.allclose(volume.affine, reference.affine, atol=1e-4, rtol=1e-4):
            raise ValueError(f"Unaligned shape/affine for {paths[name]}. Perform slice correspondence and registration upstream; this script never guesses alignment.")
        if volume.header.get_xyzt_units()[0] != reference.header.get_xyzt_units()[0]:
            raise ValueError(f"Spatial units disagree for {paths[name]}; normalize physical geometry upstream.")
    spacing = np.asarray(nib.affines.voxel_sizes(reference.affine), dtype=np.float64)
    if not np.isfinite(reference.affine).all() or not np.isfinite(spacing).all() or not (spacing > 0).all() or abs(np.linalg.det(reference.affine[:3, :3])) <= 1e-12:
        raise ValueError(f"Invalid physical geometry: {paths['bSSFP']}.")
    images = {name: normalize_image(volumes[name].get_fdata(dtype=np.float32), normalization) for name in MODALITIES}
    label = canonicalize_label(volumes["label"].get_fdata(dtype=np.float32), label_order)
    return images, label, spacing, np.asarray(reference.affine, dtype=np.float64)


def create_patient_splits(case_ids, seed=42, test_fraction=0.2, val_fraction=0.2, test_ids=None, no_val=False):
    if no_val:
        from training.dataset.no_val import REFERENCE_DIR
        from training.dataset.data_contract import read_split_names
        official = set(read_split_names(REFERENCE_DIR, "test_vol"))
        if (len(case_ids) != 380 or set(case_ids) != {f"case{i:04d}" for i in range(1, 381)}
                or test_ids is None or len(test_ids) != 76 or set(test_ids) != official):
            raise ValueError("--no-val requires the complete MyoPS380 cohort and fixed official test list")
        return {"train": sorted(set(case_ids) - official), "test_vol": sorted(official)}
    if not 0 < val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("val_fraction must be in (0,1) and test_fraction in [0,1).")
    cases = sorted(set(case_ids))
    if len(cases) != len(case_ids):
        raise ValueError("Duplicate case IDs.")
    random = np.random.default_rng(seed)
    if test_ids is None:
        count = max(1, round(len(cases) * test_fraction)) if test_fraction else 0
        if len(cases) - count < 2:
            raise ValueError("Need at least two non-test patients to form train/validation sets.")
        test = set(random.permutation(cases)[:count])
    else:
        test = set(test_ids)
        if len(test) != len(test_ids) or not test <= set(cases):
            raise ValueError("Test IDs contain duplicates or cases missing from the source dataset.")
    remaining = sorted(set(cases) - test)
    if len(remaining) < 2:
        raise ValueError("Need at least two non-test patients to form train/validation sets.")
    val_count = min(len(remaining) - 1, max(1, round(len(remaining) * val_fraction)))
    val = set(random.permutation(remaining)[:val_count])
    return {"train": sorted(set(remaining) - val), "val": sorted(val), "test_vol": sorted(test)}


def load_aligned_images(paths, normalization="unit255"):
    """Read three co-registered images without requiring a target mask.

    Return image dictionary, native spacing, affine and declared spatial unit.
    Unknown units remain unknown and must never be presented as millimetres.
    """
    if set(paths) != set(MODALITIES):
        raise ValueError(f"Expected modality paths {MODALITIES}.")
    volumes = {name: nib.load(str(path)) for name, path in paths.items()}
    reference = volumes["bSSFP"]
    if len(reference.shape) != 3 or any(size == 0 for size in reference.shape):
        raise ValueError("Expected a nonempty 3D image volume.")
    for name, volume in volumes.items():
        if volume.shape != reference.shape or not np.allclose(volume.affine, reference.affine, atol=1e-4, rtol=1e-4):
            raise ValueError(f"Unaligned shape/affine for {paths[name]}; register upstream.")
        if volume.header.get_xyzt_units()[0] != reference.header.get_xyzt_units()[0]:
            raise ValueError("Modalities use inconsistent spatial units.")
    spacing = np.asarray(nib.affines.voxel_sizes(reference.affine), dtype=np.float64)
    if not np.isfinite(reference.affine).all() or not np.isfinite(spacing).all() or not (spacing > 0).all() or abs(np.linalg.det(reference.affine[:3, :3])) <= 1e-12:
        raise ValueError("Invalid image affine or voxel spacing.")
    images = {name: normalize_image(volume.get_fdata(dtype=np.float32), normalization) for name, volume in volumes.items()}
    return images, spacing, reference.affine.copy(), reference.header.get_xyzt_units()[0]
