"""LGE Dataset — handles cached .npz files, on-the-fly preprocessing, and augmentations.

Supports:
    - LGE SAX 3D volume dataset (B, 1, D, H, W)
    - LGE LAX 2D / 2.5D slice dataset (2CH, 4CH, RAS, SAX-2D) (B, in_channels, H, W)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
import torch
from torch.utils.data import Dataset

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from preprocessing.preprocessing import preprocess_mask, preprocess_spatial

logger = logging.getLogger(__name__)


class MedicalAugmentation2D:
    """Standard 2D medical data augmentation for LGE Cardiac MRI."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_range_deg: float = 30.0,
        gamma_range: tuple[float, float] = (0.7, 1.5),
        intensity_scale: float = 0.1,
    ) -> None:
        self.flip_prob = flip_prob
        self.rotate_range_deg = rotate_range_deg
        self.gamma_range = gamma_range
        self.intensity_scale = intensity_scale

    def __call__(
        self,
        image: np.ndarray,
        label: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        # image: (C, H, W) or (H, W), label: (H, W)
        is_multi_channel = image.ndim == 3
        
        # 1. Random Flips
        if np.random.rand() < self.flip_prob:
            axis = 1 if is_multi_channel else 0
            image = np.flip(image, axis=axis)
            if label is not None:
                label = np.flip(label, axis=0)
        if np.random.rand() < self.flip_prob:
            axis = 2 if is_multi_channel else 1
            image = np.flip(image, axis=axis)
            if label is not None:
                label = np.flip(label, axis=1)

        # 2. Random Rotation
        if self.rotate_range_deg > 0:
            angle = np.random.uniform(-self.rotate_range_deg, self.rotate_range_deg)
            if is_multi_channel:
                image = rotate(image, angle, axes=(1, 2), reshape=False, order=1, mode="nearest")
            else:
                image = rotate(image, angle, reshape=False, order=1, mode="nearest")
            if label is not None:
                label = rotate(label, angle, reshape=False, order=0, mode="nearest")

        # 3. Random Gamma transform (simulates varying LGE contrast)
        if self.gamma_range:
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            image = np.clip(np.maximum(image, 0.0) ** gamma, 0.0, 1.0)

        # 4. Intensity scale and shift
        if self.intensity_scale > 0:
            scale = 1.0 + np.random.uniform(-self.intensity_scale, self.intensity_scale)
            shift = np.random.uniform(-self.intensity_scale, self.intensity_scale)
            image = np.clip(image * scale + shift, 0.0, 1.0)

        image_out = np.ascontiguousarray(image, dtype=np.float32)
        label_out = np.ascontiguousarray(label) if label is not None else None
        return image_out, label_out


class MedicalAugmentation3D:
    """Fast 3D data augmentation with in-plane rotation and contrast variation."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_range_deg: float = 20.0,
        gamma_range: tuple[float, float] = (0.7, 1.4),
        intensity_scale: float = 0.1,
    ) -> None:
        self.flip_prob = flip_prob
        self.rotate_range_deg = rotate_range_deg
        self.gamma_range = gamma_range
        self.intensity_scale = intensity_scale

    def __call__(
        self,
        image: np.ndarray,
        label: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        # image: (H, W, D), label: (H, W, D)
        if np.random.rand() < self.flip_prob:
            image = np.flip(image, axis=0)
            if label is not None:
                label = np.flip(label, axis=0)
        if np.random.rand() < self.flip_prob:
            image = np.flip(image, axis=1)
            if label is not None:
                label = np.flip(label, axis=1)

        # In-plane rotation across H, W
        if self.rotate_range_deg > 0 and np.random.rand() < 0.5:
            angle = np.random.uniform(-self.rotate_range_deg, self.rotate_range_deg)
            image = rotate(image, angle, axes=(0, 1), reshape=False, order=1, mode="nearest")
            if label is not None:
                label = rotate(label, angle, axes=(0, 1), reshape=False, order=0, mode="nearest")

        # Gamma contrast variation
        if self.gamma_range:
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            image = np.clip(np.maximum(image, 0.0) ** gamma, 0.0, 1.0)

        # Intensity scale & shift
        if self.intensity_scale > 0:
            scale = 1.0 + np.random.uniform(-self.intensity_scale, self.intensity_scale)
            shift = np.random.uniform(-self.intensity_scale, self.intensity_scale)
            image = np.clip(image * scale + shift, 0.0, 1.0)

        return np.ascontiguousarray(image, dtype=np.float32), (np.ascontiguousarray(label) if label is not None else None)


class LgeSaxDataset(Dataset):
    """LGE SAX 3D volume dataset (returns tensor of shape (1, D, H, W))."""

    def __init__(
        self,
        *,
        records: pd.DataFrame,
        data_root: Path | str,
        cache_dir: Path | str | None = None,
        target_shape: tuple[int, int, int] = (192, 192, 16),
        target_spacing: tuple[float, float, float] = (1.0, 1.0, 10.0),
        intensity_percentiles: tuple[float, float] = (0.95, 99.5),
        augment: bool = False,
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.target_shape = target_shape
        self.target_spacing = target_spacing
        self.intensity_percentiles = intensity_percentiles
        self.augmenter = MedicalAugmentation3D() if augment else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records.iloc[index]
        record_id = str(row.get("record_id", f"rec_{index}"))

        # 1. Try loading from cache if available
        if self.cache_dir and (self.cache_dir / f"{record_id}.npz").exists():
            with np.load(self.cache_dir / f"{record_id}.npz", allow_pickle=True) as data:
                image = data["image"]  # (H, W, D) = (192, 192, 16)
                label = data["label"] if "label" in data else None
        else:
            # 2. Fallback to on-the-fly reading from raw NIfTI
            raw_img_obj = nib.load(str(self.data_root / row.image_path))
            img_obj = nib.as_closest_canonical(raw_img_obj)
            raw_img = np.asanyarray(img_obj.dataobj)
            if raw_img.ndim == 4:
                raw_img = np.squeeze(raw_img)
            spacing = tuple(float(v) for v in img_obj.header.get_zooms()[:3])

            image, _ = preprocess_spatial(
                raw_img,
                source_spacing=spacing,
                target_spacing=self.target_spacing,
                target_shape=self.target_shape,
                interpolation_order=1,
                intensity_percentiles=self.intensity_percentiles,
            )
            has_label = bool(row.get("has_label", False)) and pd.notna(row.get("label_path"))
            if has_label:
                raw_lbl_obj = nib.load(str(self.data_root / row.label_path))
                lbl_obj = nib.as_closest_canonical(raw_lbl_obj)
                raw_lbl = np.rint(np.asanyarray(lbl_obj.dataobj)).astype(np.int16)
                if raw_lbl.ndim == 4:
                    raw_lbl = np.squeeze(raw_lbl)
                label = preprocess_mask(
                    raw_lbl,
                    source_spacing=spacing,
                    target_spacing=self.target_spacing,
                    target_shape=self.target_shape,
                )
            else:
                label = None

        if self.augmenter is not None:
            image, label = self.augmenter(image, label)

        # Transpose from (H, W, D) -> (D, H, W) for standard 3D Conv (B, C, D, H, W)
        image_dhw = np.transpose(image, (2, 0, 1))  # (16, 192, 192)
        sample: dict[str, Any] = {
            "record_id": record_id,
            "subject_id": str(row.get("subject_id", "")),
            "image": torch.from_numpy(image_dhw[None, ...]).float(),  # (1, D, H, W)
        }

        if label is not None:
            label_dhw = np.transpose(label, (2, 0, 1))  # (16, 192, 192)
            sample["label"] = torch.from_numpy(label_dhw).long()  # (D, H, W)

        return sample


class LgeLaxDataset(Dataset):
    """LGE 2D / 2.5D per-slice dataset (2CH, 4CH, RAS, SAX-2D).
    
    Supports:
        in_channels=1: returns (1, H, W) slice s
        in_channels=3: returns (3, H, W) 2.5D context [s-1, s, s+1]
    """

    def __init__(
        self,
        *,
        records: pd.DataFrame,
        data_root: Path | str,
        cache_dir: Path | str | None = None,
        target_shape: tuple[int, int] = (256, 256),
        target_spacing: tuple[float, float] = (1.0, 1.0),
        intensity_percentiles: tuple[float, float] | None = None,
        in_channels: int = 1,
        augment: bool = False,
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.target_shape = target_shape
        self.target_spacing = target_spacing
        self.intensity_percentiles = intensity_percentiles
        self.in_channels = in_channels
        self.augmenter = MedicalAugmentation2D() if augment else None

        # Flatten records into individual 2D slices
        self.slices: list[dict[str, Any]] = []
        for idx, row in self.records.iterrows():
            cache_file = self.cache_dir / f"{row.record_id}.npz" if self.cache_dir else None
            if cache_file and cache_file.exists():
                with np.load(cache_file, allow_pickle=True) as data:
                    d_slices = int(data["image"].shape[0]) if data["image"].ndim >= 3 else 1
            else:
                img_obj = nib.load(str(self.data_root / row.image_path))
                d_slices = int(img_obj.shape[2]) if len(img_obj.shape) >= 3 else 1

            for s in range(d_slices):
                self.slices.append({
                    "row_idx": idx,
                    "slice_idx": s,
                    "total_slices": d_slices,
                    "record_id": f"{row.record_id}_sl{s}",
                    "row": row,
                })

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.slices[index]
        row = item["row"]
        slice_idx = item["slice_idx"]
        total_slices = item.get("total_slices", 1)
        parent_rec_id = str(row.record_id)

        # 1. Try loading from cache
        if self.cache_dir and (self.cache_dir / f"{parent_rec_id}.npz").exists():
            with np.load(self.cache_dir / f"{parent_rec_id}.npz", allow_pickle=True) as data:
                all_images = data["image"]  # shape (D, H, W) or (H, W)
                all_labels = data["label"] if "label" in data else None
                
                if self.in_channels == 3 and all_images.ndim >= 3:
                    prev_idx = max(0, slice_idx - 1)
                    curr_idx = slice_idx
                    next_idx = min(total_slices - 1, slice_idx + 1)
                    image = np.stack([all_images[prev_idx], all_images[curr_idx], all_images[next_idx]], axis=0)
                elif all_images.ndim >= 3:
                    image = all_images[slice_idx][None, ...]  # (1, H, W)
                else:
                    image = all_images[None, ...]
                    if self.in_channels == 3:
                        image = np.repeat(image, 3, axis=0)

                label = all_labels[slice_idx] if all_labels is not None and all_labels.ndim >= 3 else (all_labels if all_labels is not None else None)
        else:
            # 2. Fallback to on-the-fly NIfTI loading
            raw_img_obj = nib.load(str(self.data_root / row.image_path))
            img_obj = nib.as_closest_canonical(raw_img_obj)
            raw_img = np.asanyarray(img_obj.dataobj)
            spacing = tuple(float(v) for v in img_obj.header.get_zooms()[:2])

            if self.in_channels == 3 and raw_img.ndim >= 3:
                prev_idx = max(0, slice_idx - 1)
                curr_idx = slice_idx
                next_idx = min(total_slices - 1, slice_idx + 1)
                slices_data = [raw_img[:, :, prev_idx], raw_img[:, :, curr_idx], raw_img[:, :, next_idx]]
                processed_channels = []
                for s_data in slices_data:
                    p_img, _ = preprocess_spatial(
                        s_data,
                        source_spacing=spacing,
                        target_spacing=self.target_spacing,
                        target_shape=self.target_shape,
                        interpolation_order=1,
                        intensity_percentiles=self.intensity_percentiles,
                    )
                    processed_channels.append(p_img)
                image = np.stack(processed_channels, axis=0)  # (3, H, W)
            else:
                slice_data = raw_img[:, :, slice_idx] if raw_img.ndim >= 3 else raw_img
                p_img, _ = preprocess_spatial(
                    slice_data,
                    source_spacing=spacing,
                    target_spacing=self.target_spacing,
                    target_shape=self.target_shape,
                    interpolation_order=1,
                    intensity_percentiles=self.intensity_percentiles,
                )
                image = p_img[None, ...]
                if self.in_channels == 3:
                    image = np.repeat(image, 3, axis=0)

            has_label = bool(row.get("has_label", False)) and pd.notna(row.get("label_path"))
            if has_label:
                raw_lbl_obj = nib.load(str(self.data_root / row.label_path))
                lbl_obj = nib.as_closest_canonical(raw_lbl_obj)
                raw_lbl = np.rint(np.asanyarray(lbl_obj.dataobj)).astype(np.int16)
                lbl_slice = raw_lbl[:, :, slice_idx] if raw_lbl.ndim >= 3 else raw_lbl
                label = preprocess_mask(
                    lbl_slice,
                    source_spacing=spacing,
                    target_spacing=self.target_spacing,
                    target_shape=self.target_shape,
                )
            else:
                label = None

        if self.augmenter is not None:
            image, label = self.augmenter(image, label)

        sample: dict[str, Any] = {
            "record_id": item["record_id"],
            "subject_id": str(row.get("subject_id", "")),
            "image": torch.from_numpy(image).float(),
        }

        if label is not None:
            sample["label"] = torch.from_numpy(label).long()

        return sample

