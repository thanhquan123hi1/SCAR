"""Inference entry point: load checkpoint and save segmentation masks.

Usage:
    python training/predict.py \
        --config outputs/runs/unet3d_lge_sax_exp01/config_snapshot.yaml \
        --checkpoint outputs/runs/unet3d_lge_sax_exp01/checkpoints/best.pt \
        --split validation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.models import build_model
from preprocessing.preprocessing import (
    extract_tissue_foreground,
    invert_spatial_mask,
    preprocess_spatial,
)
from training.postprocess import decode_with_rules, enforce_anatomical_constraints

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LGE segmentation inference.")
    parser.add_argument("--config", required=True, help="Config snapshot YAML")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint .pt")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--output", default=None, help="Output directory (default: run_dir/predictions)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    num_classes = int(config.get("num_classes", 5))
    model = build_model(config["model_name"], **{**config.get("model", {}), "num_classes": num_classes})
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    logger.info("Loaded checkpoint from epoch %d", ckpt.get("epoch", -1))

    # Data
    splits_dir = ROOT / config["data"]["splits_dir"]
    df = pd.read_csv(splits_dir / f"{args.split}.csv")
    view = str(config.get("view", "SAX")).upper()
    df = df[df["view"] == view]
    raw_root = ROOT / config["data"]["raw_root"]

    pp = config.get("preprocessing", {})
    target_shape = tuple(pp["target_shape"])
    target_spacing = tuple(pp["target_spacing"])
    percentiles = tuple(pp["intensity_percentiles"]) if pp.get("intensity_percentiles") else None

    post_cfg = config.get("postprocess", {})
    use_rules = post_cfg.get("use_rules", True)
    in_channels = int(config.get("training", {}).get("in_channels", config.get("model", {}).get("in_channels", 1)))

    # Output dir
    run_dir = Path(args.config).parent
    out_dir = Path(args.output) if args.output else run_dir / "predictions" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        raw_nii_obj = nib.load(str(raw_root / row.image_path))
        nii_obj = nib.as_closest_canonical(raw_nii_obj)
        raw_image = np.asanyarray(nii_obj.dataobj)
        zooms = nii_obj.header.get_zooms()

        # --- CASE 1: 3D SAX ---
        if view == "SAX" and len(target_shape) == 3:
            if raw_image.ndim == 4:
                if raw_image.shape[-1] == 1:
                    raw_image = raw_image[..., 0]
                    zooms = zooms[:3]
                elif raw_image.shape[0] == 1:
                    raw_image = raw_image[0, ...]
                    zooms = zooms[1:4]
                else:
                    raw_image = raw_image[:, :, :, 0]
                    zooms = zooms[:3]

            spacing = tuple(float(v) for v in zooms[:3])
            processed, transform = preprocess_spatial(
                raw_image,
                source_spacing=spacing,
                target_spacing=target_spacing,
                target_shape=target_shape,
                interpolation_order=1,
                intensity_percentiles=percentiles,
            )
            proc_dhw = np.transpose(processed, (2, 0, 1))  # (D, H, W)
            tensor = torch.from_numpy(proc_dhw[None, None, ...]).to(device, dtype=torch.float32)

            with torch.no_grad():
                logits = model(tensor)
                if logits.shape[1] == num_classes - 1:
                    pred_dhw = decode_with_rules(logits, view=view)[0].cpu().numpy().astype(np.int16)
                elif use_rules and logits.shape[1] == num_classes:
                    pred_dhw = decode_with_rules(logits[:, 1:], view=view)[0].cpu().numpy().astype(np.int16)
                else:
                    pred_dhw = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int16)

            pred_processed = np.transpose(pred_dhw, (1, 2, 0))  # (H, W, D)
            restored_pred = invert_spatial_mask(pred_processed, transform)

        # --- CASE 2: 2D LAX (2CH, 4CH, RAS) ---
        else:
            if raw_image.ndim == 4:
                if raw_image.shape[-1] == 1:
                    raw_image = raw_image[..., 0]
                    zooms = zooms[:3]
                elif raw_image.shape[0] == 1:
                    raw_image = raw_image[0, ...]
                    zooms = zooms[1:4]
                else:
                    raw_image = raw_image[:, :, :, 0]
                    zooms = zooms[:3]

            spacing = tuple(float(v) for v in zooms[:2])
            d_count = raw_image.shape[2] if raw_image.ndim >= 3 else 1
            restored_slices = []

            # Global foreground intensity bounds across full volume (fixes C1)
            precomputed_bounds = None
            if percentiles is not None:
                full_vol = np.asarray(raw_image, dtype=np.float32)
                fg = extract_tissue_foreground(full_vol)
                p_low, p_high = np.percentile(fg, percentiles)
                if np.isfinite(p_low) and np.isfinite(p_high) and p_high > p_low:
                    precomputed_bounds = (float(p_low), float(p_high))

            for s in range(d_count):
                if in_channels == 3 and raw_image.ndim >= 3:
                    # Edge clamping on boundaries (preserves through-plane gradient)
                    prev_idx = max(0, s - 1)
                    curr_idx = s
                    next_idx = min(d_count - 1, s + 1)
                    slices_data = [raw_image[:, :, prev_idx], raw_image[:, :, curr_idx], raw_image[:, :, next_idx]]
                    processed_channels = []
                    trans_s = None
                    for s_data in slices_data:
                        p_img, t_obj = preprocess_spatial(
                            s_data,
                            source_spacing=spacing,
                            target_spacing=target_spacing,
                            target_shape=target_shape,
                            interpolation_order=1,
                            intensity_percentiles=None if precomputed_bounds is not None else percentiles,
                            precomputed_intensity_bounds=precomputed_bounds,
                        )
                        processed_channels.append(p_img)
                        if trans_s is None:
                            trans_s = t_obj
                    stacked = np.stack(processed_channels, axis=0)
                    t_sl = torch.from_numpy(stacked[None, ...]).to(device, dtype=torch.float32)
                else:
                    slice_img = raw_image[:, :, s] if raw_image.ndim >= 3 else raw_image
                    p_sl, trans_s = preprocess_spatial(
                        slice_img,
                        source_spacing=spacing,
                        target_spacing=target_spacing,
                        target_shape=target_shape,
                        interpolation_order=1,
                        intensity_percentiles=None if precomputed_bounds is not None else percentiles,
                        precomputed_intensity_bounds=precomputed_bounds,
                    )
                    stacked = p_sl[None, ...]
                    if in_channels == 3:
                        stacked = np.repeat(stacked, 3, axis=0)
                    t_sl = torch.from_numpy(stacked[None, ...]).to(device, dtype=torch.float32)

                with torch.no_grad():
                    logits = model(t_sl)
                    if logits.shape[1] == num_classes - 1:
                        p_out = decode_with_rules(logits, view=view)[0].cpu().numpy().astype(np.int16)
                    elif use_rules and logits.shape[1] == num_classes:
                        p_out = decode_with_rules(logits[:, 1:], view=view)[0].cpu().numpy().astype(np.int16)
                    else:
                        p_out = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int16)

                rest_s = invert_spatial_mask(p_out, trans_s)
                restored_slices.append(rest_s)

            restored_pred = np.stack(restored_slices, axis=-1) if raw_image.ndim >= 3 else restored_slices[0]

        # Apply anatomical constraints (spacing & physical volume aware)
        if post_cfg.get("anatomical_constraint", True):
            full_spacing = tuple(float(v) for v in zooms[:restored_pred.ndim])
            restored_pred = enforce_anatomical_constraints(
                restored_pred,
                scar_class=3,
                myo_class=2,
                dilation_voxels=int(post_cfg.get("dilation_voxels", 1)),
                tolerance_mm=float(post_cfg.get("tolerance_mm", 2.5)),
                spacing=full_spacing,
                min_scar_voxels=int(post_cfg.get("min_scar_voxels", 5)),
                min_scar_volume_mm3=float(post_cfg.get("min_scar_volume_mm3", 15.0)),
            )

        # Save restored NIfTI prediction
        mask = restored_pred
        out_nii = nib.Nifti1Image(mask.astype(np.uint8), nii_obj.affine, nii_obj.header)
        out_path = out_dir / Path(row.image_path).name
        nib.save(out_nii, str(out_path))
        logger.info("Saved: %s", out_path.name)

    logger.info("Predictions saved to %s", out_dir)


if __name__ == "__main__":
    main()
