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
from preprocessing.preprocessing import preprocess_spatial, invert_spatial_mask

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
    df = df[df["view"] == config.get("view", "SAX")]
    raw_root = ROOT / config["data"]["raw_root"]

    pp = config.get("preprocessing", {})
    target_shape = tuple(pp["target_shape"])
    target_spacing = tuple(pp["target_spacing"])
    percentiles = tuple(pp["intensity_percentiles"]) if pp.get("intensity_percentiles") else None

    # Output dir
    run_dir = Path(args.config).parent
    out_dir = Path(args.output) if args.output else run_dir / "predictions" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        nii_obj = nib.load(str(raw_root / row.image_path))
        image = np.asanyarray(nii_obj.dataobj)
        spacing = tuple(float(v) for v in nii_obj.header.get_zooms()[:len(target_shape)])

        processed, transform = preprocess_spatial(
            image,
            source_spacing=spacing,
            target_spacing=target_spacing,
            target_shape=target_shape,
            interpolation_order=1,
            intensity_percentiles=percentiles,
        )
        with torch.no_grad():
            tensor = torch.from_numpy(processed[None, None, ...]).to(device, dtype=torch.float32)
            logits = model(tensor)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int16)

        # Restore to original geometry
        mask = invert_spatial_mask(pred, transform)
        out_nii = nib.Nifti1Image(mask.astype(np.uint8), nii_obj.affine, nii_obj.header)
        out_path = out_dir / Path(row.image_path).name
        nib.save(out_nii, str(out_path))
        logger.info("Saved: %s", out_path.name)

    logger.info("Predictions saved to %s", out_dir)


if __name__ == "__main__":
    main()
