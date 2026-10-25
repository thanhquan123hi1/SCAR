"""Aligned CINE/LGE/T2w input dataset with a single canonical four-class target.

Slice files contain [H,W] arrays; volume files contain [H,W,D] arrays.
Normalize before augmentation; spatial transforms stay synchronized.
"""
from __future__ import annotations

from pathlib import Path
import random
import re

import h5py
import numpy as np
from scipy import ndimage
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from training.dataset.data_contract import (
    canonicalize_label,
    patient_id,
    read_split_names,
    resolve_label_order,
)


def random_rot_flip(image, image1, image2, label):
    k, axis = np.random.randint(0, 4), np.random.randint(0, 2)
    return tuple(np.flip(np.rot90(x, k), axis=axis).copy() for x in (image, image1, image2, label))


def random_rotate(image, image1, image2, label):
    angle = np.random.uniform(-20, 20)
    return tuple(
        ndimage.rotate(x, angle, order=0 if i == 3 else 1, reshape=False, mode="constant", cval=0, prefilter=False)
        for i, x in enumerate((image, image1, image2, label))
    )


def add_gaussian_noise(image, image1, image2, label, mean=0, std=0.02):
    images = tuple(np.clip(x + np.random.normal(mean, std, x.shape), 0, 1) for x in (image, image1, image2))
    return (*images, label)


class ResizeGenerator:
    """Deterministic, alignment-preserving image resize and tensor conversion."""

    def __init__(self, output_size):
        if len(output_size) != 2 or any(int(v) <= 0 for v in output_size):
            raise ValueError("output_size must contain two positive dimensions.")
        self.output_size = tuple(int(v) for v in output_size)

    def __call__(self, sample):
        result = dict(sample)
        shape = sample["label"].shape
        if len(shape) != 2:
            raise ValueError("Slice transforms require 2D samples; volume resize belongs in inference.")
        for key in ("image", "image1", "image2", "label"):
            value = np.asarray(sample[key])
            if value.shape != shape:
                raise ValueError("Modalities and mask must have identical spatial shapes.")
            dtype = np.int64 if key == "label" else np.float32
            tensor = torch.from_numpy(np.array(value, dtype=dtype, copy=True, order="C"))
            if shape != self.output_size:
                if key == "label":
                    tensor = F.interpolate(tensor[None, None].float(), size=self.output_size, mode="nearest")[0, 0].long()
                else:
                    tensor = F.interpolate(tensor[None, None], size=self.output_size, mode="bilinear", align_corners=False)[0, 0]
            result[key] = tensor if key == "label" else tensor.unsqueeze(0)
        return result


class RandomGenerator(ResizeGenerator):
    """Shared spatial warps and independent intensity variation on [0,1] images.

    Translation pads with background and never wraps structures at image edges.
    """

    def __call__(self, sample):
        result = dict(sample)
        images = [np.asarray(sample[k]).copy() for k in ("image", "image1", "image2")]
        label = np.asarray(sample["label"]).copy()
        if random.random() < 0.5:
            *images, label = random_rot_flip(*images, label)
        elif random.random() < 0.5:
            *images, label = random_rotate(*images, label)
        if random.random() < 0.5:
            shift = (random.randint(-5, 5), random.randint(-5, 5))
            images = [ndimage.shift(x, shift, order=1, mode="constant", cval=0, prefilter=False) for x in images]
            label = ndimage.shift(label, shift, order=0, mode="constant", cval=0, prefilter=False)
        for i, image in enumerate(images):
            if random.random() < 0.5:
                image = np.clip(image * random.uniform(0.8, 1.2), 0, 1)
            if random.random() < 0.5:
                image = np.clip(image, 0, 1) ** random.uniform(0.8, 1.2)
            if random.random() < 0.25:
                image = np.clip(image + np.random.normal(0, 0.02, image.shape), 0, 1)
            result[("image", "image1", "image2")[i]] = image
        result["label"] = label
        return super().__call__(result)


class MyopsDataset(Dataset):
    def __init__(self, base_dir, base_dir1, base_dir2, list_dir, split, transform=None, label_order="auto", validate_files=True):
        self.transform = transform
        self.split = split
        self.sample_list = read_split_names(list_dir, split)
        if not self.sample_list:
            raise ValueError(f"The {split} dataset is empty.")
        self.data_dir, self.data_dir1, self.data_dir2 = map(Path, (base_dir, base_dir1, base_dir2))
        self.roots = (self.data_dir, self.data_dir1, self.data_dir2)
        self.label_order = label_order if label_order == "auto" else resolve_label_order(label_order)
        self.is_volume = split.endswith("vol") or split in {"test", "volume"}
        self.extension = ".npy.h5" if self.is_volume else ".npz"
        self.volume_slices = {}
        if split == "val_vol":
            # The original release has validation NPZ slices but no val HDF5.
            # Reassemble complete patients read-only, never invent geometry.
            for name in self.sample_list:
                volumes = [root / f"{name}.npy.h5" for root in self.roots]
                if all(path.is_file() for path in volumes):
                    continue
                if any(path.is_file() for path in volumes):
                    raise FileNotFoundError(f"Incomplete validation volumes for {name}")
                paths = list((self.roots[0].parent / "train_npz").glob(f"{name}_slice*.npz"))
                indexed = []
                for path in paths:
                    match = re.fullmatch(re.escape(name) + r"_slice(\d+)", path.stem)
                    if match:
                        indexed.append((int(match[1]), path.name))
                indexed.sort()
                if not indexed or [i for i, _ in indexed] != list(range(len(indexed))):
                    raise ValueError(f"Validation slices must be complete and contiguous from zero: {name}")
                self.volume_slices[name] = [filename for _, filename in indexed]
        if validate_files:
            required = [root.parent / "train_npz" / filename
                        for name, filenames in self.volume_slices.items()
                        for root in self.roots for filename in filenames]
            required += [root / f"{name}{self.extension}" for name in self.sample_list
                         if name not in self.volume_slices for root in self.roots]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} aligned modality file(s); first: {missing[:5]}")

    def __len__(self):
        return len(self.sample_list)

    def _read(self, path):
        if path.name.endswith(".h5"):
            with h5py.File(path, "r") as data:
                image = data["image"][:]
                label = data["label"][:] if "label" in data else None
                metadata = dict(data.attrs)
                for key in ("spacing", "affine"):
                    if key in data:
                        metadata[key] = data[key][:]
        else:
            with np.load(path, allow_pickle=False) as data:
                image = data["image"].copy()
                label = data["label"].copy() if "label" in data else None
                metadata = {key: data[key].copy() for key in ("label_order", "spacing", "affine", "spacing_unit", "patient_id", "axis_order", "normalization") if key in data}
        stored_order = metadata.get("label_order")
        if stored_order is None:
            if self.label_order == "auto":
                raise ValueError(f"Missing label_order metadata in {path}; specify legacy or canonical explicitly.")
            order = self.label_order
        else:
            order = resolve_label_order(stored_order)
            if self.label_order != "auto" and order != self.label_order:
                raise ValueError(f"Requested label order conflicts with metadata in {path}.")
        if label is not None:
            label = canonicalize_label(label, order)
        return np.asarray(image, dtype=np.float32), label, metadata

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        if name in self.volume_slices:
            modalities = []
            for root in self.roots:
                pieces = [self._read(root.parent / "train_npz" / filename)
                          for filename in self.volume_slices[name]]
                first = pieces[0][2]
                for _, target, meta in pieces:
                    if target is None:
                        raise ValueError(f"Missing target in validation slices for {name}")
                    for key in ("affine", "spacing", "spacing_unit", "patient_id", "label_order", "normalization"):
                        if (key in first) != (key in meta) or (key in first and not np.array_equal(first[key], meta[key])):
                            raise ValueError(f"Inconsistent slice {key} metadata for {name}")
                metadata = dict(first, axis_order="HWD")
                modalities.append((np.stack([p[0] for p in pieces], axis=2),
                                   np.stack([p[1] for p in pieces], axis=2), metadata))
        else:
            modalities = [self._read(root / f"{name}{self.extension}") for root in self.roots]
        image, label, metadata = modalities[0]
        if label is None:
            raise ValueError(f"Missing target label in CINE file for {name}.")
        expected_ndim = 3 if self.is_volume else 2
        if image.ndim != expected_ndim or image.shape != label.shape or any(size == 0 for size in image.shape):
            raise ValueError(f"Invalid {name} image/label shape: {image.shape}/{label.shape}; expected {expected_ndim}D aligned arrays.")
        for other_image, other_label, other_metadata in modalities:
            if other_image.shape != image.shape or not np.isfinite(other_image).all():
                raise ValueError(f"Non-finite or misaligned modality data for {name}.")
            if other_image.min() < -1e-5 or other_image.max() > 1 + 1e-5:
                raise ValueError(f"Images must be normalized to [0,1] in preprocessing; out-of-range intensity in {name}.")
            if other_label is None or not np.array_equal(label, other_label):
                raise ValueError(f"Modality target masks differ for {name}; align modalities and use one shared label.")
            if "patient_id" in other_metadata and str(np.asarray(other_metadata["patient_id"]).item()) != patient_id(name):
                raise ValueError(f"Patient identity in metadata disagrees with filename {name}.")
            if str(np.asarray(metadata.get("spacing_unit", "unknown")).item()) != str(np.asarray(other_metadata.get("spacing_unit", "unknown")).item()):
                raise ValueError(f"Modalities use inconsistent spacing units for {name}.")
            for key in ("spacing", "affine"):
                if (key in metadata) != (key in other_metadata):
                    raise ValueError(f"Inconsistent {key} metadata between modalities for {name}.")
                if key in metadata and not np.allclose(metadata[key], other_metadata[key], atol=1e-4, rtol=1e-4):
                    raise ValueError(f"Modalities have incompatible {key} for {name}; registration is required upstream.")
            if "axis_order" in other_metadata and str(np.asarray(other_metadata["axis_order"]).item()) != ("HWD" if self.is_volume else "HW"):
                raise ValueError(f"Unsupported axis_order for {name}; expected HWD volumes or HW slices.")
        spacing = np.asarray(metadata.get("spacing", [np.nan] * 3), dtype=np.float64)
        affine = np.asarray(metadata.get("affine", np.full((4, 4), np.nan)), dtype=np.float64)
        valid_geometry = bool(spacing.shape == (3,) and affine.shape == (4, 4) and np.isfinite(spacing).all() and (spacing > 0).all() and np.isfinite(affine).all() and abs(np.linalg.det(affine[:3, :3])) > 1e-12)
        if ("spacing" in metadata or "affine" in metadata) and not valid_geometry:
            raise ValueError(f"Invalid physical geometry metadata for {name}.")
        spacing_unit = str(np.asarray(metadata.get("spacing_unit", "unknown")).item())
        if spacing_unit not in {"mm", "unknown"}:
            raise ValueError(f"Unsupported spacing unit {spacing_unit!r}; preprocess to mm first.")
        if valid_geometry and not np.allclose(np.linalg.norm(affine[:3, :3], axis=0), spacing, atol=1e-4, rtol=1e-4):
            raise ValueError(f"Spacing and affine voxel sizes disagree for {name}.")
        has_geometry = valid_geometry and spacing_unit == "mm"
        sample = {
            "image": image,
            "image1": modalities[1][0],
            "image2": modalities[2][0],
            "label": label.astype(np.int64),
            "case_name": name,
            "patient_id": patient_id(name),
            "spacing": spacing,
            "affine": affine,
            "has_geometry": has_geometry,
            "has_affine": valid_geometry,
            "spacing_unit": spacing_unit,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


Myops_dataset = MyopsDataset
MultimodalCMRDataset = MyopsDataset
