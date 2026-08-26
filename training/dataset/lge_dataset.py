"""LGE Dataset — handles cached .npz files, on-the-fly preprocessing, and augmentations.

Supports:
    - LGE SAX 3D volume dataset (B, 1, D, H, W)
    - LGE LAX 2D slice dataset (2CH, 4CH, RAS) (B, 1, H, W)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from preprocessing.preprocessing import preprocess_mask, preprocess_spatial


class MedicalAugmentation3D:
    """Fast in-memory 3D data augmentation."""

    def __init__(self, flip_prob: float = 0.5, intensity_scale: float = 0.1) -> None:
        self.flip_prob = flip_prob
        self.intensity_scale = intensity_scale

    def __call__(self, image: np.ndarray, label: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        # image: (H, W, D), label: (H, W, D)
        if np.random.rand() < self.flip_prob:
            image = np.flip(image, axis=0)
            if label is not None:
                label = np.flip(label, axis=0)
        if np.random.rand() < self.flip_prob:
            image = np.flip(image, axis=1)
            if label is not None:
                label = np.flip(label, axis=1)
        if self.intensity_scale > 0:
            scale = 1.0 + np.random.uniform(-self.intensity_scale, self.intensity_scale)
            shift = np.random.uniform(-self.intensity_scale, self.intensity_scale)
            image = np.clip(image * scale + shift, 0.0, 1.0)

        return np.ascontiguousarray(image), (np.ascontiguousarray(label) if label is not None else None)


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
            data = np.load(self.cache_dir / f"{record_id}.npz", allow_pickle=True)
            image = data["image"]  # (H, W, D) = (192, 192, 16)
            label = data["label"] if "label" in data else None
        else:
            # 2. Fallback to on-the-fly reading from raw NIfTI
            img_obj = nib.load(str(self.data_root / row.image_path))
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
                lbl_obj = nib.load(str(self.data_root / row.label_path))
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
    """LGE LAX 2D per-slice dataset (2CH, 4CH, RAS) (returns tensor of shape (1, H, W))."""

    def __init__(
        self,
        *,
        records: pd.DataFrame,
        data_root: Path | str,
        cache_dir: Path | str | None = None,
        target_shape: tuple[int, int] = (256, 256),
        target_spacing: tuple[float, float] = (1.0, 1.0),
        intensity_percentiles: tuple[float, float] | None = None,
        augment: bool = False,
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.target_shape = target_shape
        self.target_spacing = target_spacing
        self.intensity_percentiles = intensity_percentiles

        # Flatten records into individual 2D slices
        self.slices: list[dict[str, Any]] = []
        for idx, row in self.records.iterrows():
            img_obj = nib.load(str(self.data_root / row.image_path))
            d_slices = int(img_obj.shape[2]) if len(img_obj.shape) >= 3 else 1
            for s in range(d_slices):
                self.slices.append({
                    "row_idx": idx,
                    "slice_idx": s,
                    "record_id": f"{row.record_id}_sl{s}",
                    "row": row,
                })

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.slices[index]
        row = item["row"]
        slice_idx = item["slice_idx"]
        parent_rec_id = str(row.record_id)

        # 1. Try loading from cache
        if self.cache_dir and (self.cache_dir / f"{parent_rec_id}.npz").exists():
            data = np.load(self.cache_dir / f"{parent_rec_id}.npz", allow_pickle=True)
            image = data["image"][slice_idx]  # (256, 256)
            label = data["label"][slice_idx] if "label" in data else None
        else:
            # 2. Fallback to on-the-fly NIfTI loading
            img_obj = nib.load(str(self.data_root / row.image_path))
            raw_img = np.asanyarray(img_obj.dataobj)
            spacing = tuple(float(v) for v in img_obj.header.get_zooms()[:2])

            slice_data = raw_img[:, :, slice_idx] if raw_img.ndim >= 3 else raw_img
            image, _ = preprocess_spatial(
                slice_data,
                source_spacing=spacing,
                target_spacing=self.target_spacing,
                target_shape=self.target_shape,
                interpolation_order=1,
                intensity_percentiles=self.intensity_percentiles,
            )

            has_label = bool(row.get("has_label", False)) and pd.notna(row.get("label_path"))
            if has_label:
                lbl_obj = nib.load(str(self.data_root / row.label_path))
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

        sample: dict[str, Any] = {
            "record_id": item["record_id"],
            "subject_id": str(row.get("subject_id", "")),
            "image": torch.from_numpy(image[None, ...]).float(),  # (1, H, W)
        }

        if label is not None:
            sample["label"] = torch.from_numpy(label).long()  # (H, W)

        return sample
