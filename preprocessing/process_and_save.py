"""Offline preprocessing script for LGE CMR-MULTI dataset.

Processes raw NIfTI files into cached compressed .npz archives:
- SAX: 3D volume (192, 192, 16) with spacing (1.0, 1.0, 10.0) mm
- LAX (2CH, 4CH, RAS): 2D slices (256, 256) with spacing (1.0, 1.0) mm

Usage:
    python preprocessing/process_and_save.py \
        --data-root data/LGE_MULTI \
        --splits-dir data/processed/splits \
        --output-dir data/processed/cache \
        --config preprocessing/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.preprocessing import preprocess_mask, preprocess_spatial

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_manifest(
    *,
    df: pd.DataFrame,
    data_root: Path,
    output_dir: Path,
    config: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_sax = config["lge_preprocessing"]["sax"]
    cfg_lax = config["lge_preprocessing"]["lax"]

    saved_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Caching LGE Data"):
        view = str(row["view"]).upper()
        split = str(row["split"])
        subject_id = str(row["subject_id"])
        record_id = str(row["record_id"])

        img_file = data_root / row["image_path"]
        if not img_file.exists():
            logger.warning("Image file not found: %s", img_file)
            continue

        raw_nii_img = nib.load(str(img_file))
        nii_img = nib.as_closest_canonical(raw_nii_img)
        image_data = np.asanyarray(nii_img.dataobj)
        affine = nii_img.affine
        zooms = nii_img.header.get_zooms()

        has_label = bool(row.get("has_label", False)) and pd.notna(row.get("label_path"))
        lbl_data = None
        if has_label:
            lbl_file = data_root / row["label_path"]
            if lbl_file.exists():
                raw_nii_lbl = nib.load(str(lbl_file))
                nii_lbl = nib.as_closest_canonical(raw_nii_lbl)
                lbl_data = np.rint(np.asanyarray(nii_lbl.dataobj)).astype(np.int16)

        save_dict: dict = {
            "affine": affine,
            "view": view,
            "split": split,
            "subject_id": subject_id,
            "has_label": np.array(has_label),
        }

        # ---------------------------------------------------------------
        # CASE 1: SAX (3D Volume)
        # ---------------------------------------------------------------
        if view == "SAX":
            if image_data.ndim == 4:
                image_data = image_data[..., 0] if image_data.shape[-1] == 1 else image_data[:, :, :, 0]
            if lbl_data is not None and lbl_data.ndim == 4:
                lbl_data = lbl_data[..., 0] if lbl_data.shape[-1] == 1 else lbl_data[:, :, :, 0]

            target_shape = tuple(cfg_sax["target_shape"])       # (192, 192, 16)
            target_spacing = tuple(cfg_sax["target_spacing"])   # (1.0, 1.0, 10.0)
            percentiles = (
                tuple(cfg_sax["intensity_percentiles"])
                if cfg_sax.get("intensity_percentiles")
                else None
            )
            orig_spacing = tuple(float(v) for v in zooms[:3])

            proc_image, transform = preprocess_spatial(
                image_data,
                source_spacing=orig_spacing,
                target_spacing=target_spacing,
                target_shape=target_shape,
                interpolation_order=1,
                intensity_percentiles=percentiles,
            )
            save_dict["image"] = proc_image.astype(np.float32)
            save_dict["transform_json"] = json.dumps(transform.to_dict())

            if lbl_data is not None:
                proc_label = preprocess_mask(
                    lbl_data,
                    source_spacing=orig_spacing,
                    target_spacing=target_spacing,
                    target_shape=target_shape,
                )
                save_dict["label"] = proc_label.astype(np.int16)

        # ---------------------------------------------------------------
        # CASE 2: LAX (2CH, 4CH, RAS - 2D Slices)
        # ---------------------------------------------------------------
        else:
            target_shape = tuple(cfg_lax["target_shape"])       # (256, 256)
            target_spacing = tuple(cfg_lax["target_spacing"])   # (1.0, 1.0)
            percentiles = (
                tuple(cfg_lax["intensity_percentiles"])
                if cfg_lax.get("intensity_percentiles")
                else None
            )
            orig_spacing_2d = tuple(float(v) for v in zooms[:2])

            # BUG #1 FIX: Compute percentile bounds on the FULL 3D volume ONCE,
            # then pass to each slice for consistent normalization.
            precomputed_bounds = None
            if percentiles is not None:
                full_vol = np.asarray(image_data, dtype=np.float32)
                p_low, p_high = np.percentile(full_vol, percentiles)
                if np.isfinite(p_low) and np.isfinite(p_high) and p_high > p_low:
                    precomputed_bounds = (float(p_low), float(p_high))

            num_slices = image_data.shape[2] if image_data.ndim >= 3 else 1
            proc_slices = []
            proc_labels = []
            transforms = []

            for z in range(num_slices):
                slice_img = image_data[:, :, z] if image_data.ndim >= 3 else image_data
                p_img, trans = preprocess_spatial(
                    slice_img,
                    source_spacing=orig_spacing_2d,
                    target_spacing=target_spacing,
                    target_shape=target_shape,
                    interpolation_order=1,
                    intensity_percentiles=percentiles if precomputed_bounds is None else None,
                    precomputed_intensity_bounds=precomputed_bounds,
                )
                proc_slices.append(p_img)
                transforms.append(trans.to_dict())

                if lbl_data is not None:
                    slice_lbl = lbl_data[:, :, z] if lbl_data.ndim >= 3 else lbl_data
                    p_lbl = preprocess_mask(
                        slice_lbl,
                        source_spacing=orig_spacing_2d,
                        target_spacing=target_spacing,
                        target_shape=target_shape,
                    )
                    proc_labels.append(p_lbl)

            save_dict["image"] = np.stack(proc_slices, axis=0).astype(np.float32)  # (D, 256, 256)
            save_dict["transform_json"] = json.dumps(transforms)
            if proc_labels:
                save_dict["label"] = np.stack(proc_labels, axis=0).astype(np.int16)  # (D, 256, 256)

        save_path = output_dir / f"{record_id}.npz"
        np.savez_compressed(save_path, **save_dict)
        saved_count += 1

    logger.info("Successfully cached %d LGE records into %s", saved_count, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache preprocessed LGE studies.")
    parser.add_argument("--data-root", default="data/LGE_MULTI", help="Path to LGE data root")
    parser.add_argument("--splits-dir", default="data/processed/splits", help="Directory with CSV splits")
    parser.add_argument("--output-dir", default="data/processed/cache", help="Output directory for npz files")
    parser.add_argument("--config", default="preprocessing/config.yaml", help="Preprocessing config YAML")
    args = parser.parse_args()

    data_root = ROOT / args.data_root
    splits_dir = ROOT / args.splits_dir
    output_dir = ROOT / args.output_dir
    config_path = ROOT / args.config

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    manifest_file = splits_dir / "manifest.csv"
    if not manifest_file.exists():
        logger.info("Manifest not found at %s. Running build_splits first...", manifest_file)
        import subprocess
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "preprocessing/build_splits.py"),
                "--data-root",
                str(data_root),
                "--output",
                str(splits_dir),
            ],
            check=True,
        )

    df = pd.read_csv(manifest_file)
    logger.info("Starting offline caching for %d records...", len(df))
    process_manifest(
        df=df,
        data_root=data_root,
        output_dir=output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
