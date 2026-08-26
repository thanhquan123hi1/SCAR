"""Comprehensive evaluation script for LGE segmentation and clinical scar quantification.

Supports:
- Multi-view evaluation: SAX (3D volume), 2CH (2D slices), 4CH (2D slices), RAS (2D slices)
- Per-class Dice score and IoU
- Physical-space Hausdorff Distance (HD95 mm)
- Clinical scar metrics (for SAX/2CH/4CH): Scar volume (mL), Myocardial Scar mass (g), Scar percentage (%)
- Exports per_subject_metrics.csv and per_class_summary.csv

Usage:
    python training/evaluate.py \
        --config outputs/runs/unet3d_lge_sax_exp01/config_snapshot.yaml \
        --checkpoint outputs/runs/unet3d_lge_sax_exp01/checkpoints/best.pt \
        --split validation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.preprocessing import invert_spatial_mask, preprocess_mask, preprocess_spatial
from training.metrics import calculate_scar_metrics, dice_score, hd95_binary, iou_score
from training.models import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

VIEW_LABEL_NAMES = {
    "SAX": {0: "background", 1: "lv_cavity", 2: "lv_myo", 3: "scar", 4: "rv_cavity"},
    "4CH": {0: "background", 1: "lv_cavity", 2: "lv_myo", 3: "scar", 4: "rv_cavity"},
    "2CH": {0: "background", 1: "lv_cavity", 2: "lv_myo", 3: "scar"},
    "RAS": {0: "background", 1: "right_atrium"},
}


def evaluate_split(
    *,
    model: torch.nn.Module,
    df: pd.DataFrame,
    data_root: Path,
    config: dict,
    device: torch.device,
    save_predictions: bool = True,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_classes = int(config.get("num_classes", 5))
    view = str(config.get("view", "SAX")).upper()
    pp = config.get("preprocessing", {})
    target_shape = tuple(pp["target_shape"])
    target_spacing = tuple(pp["target_spacing"])
    percentiles = tuple(pp["intensity_percentiles"]) if pp.get("intensity_percentiles") else None

    label_names = VIEW_LABEL_NAMES.get(view, {i: f"class_{i}" for i in range(num_classes)})
    per_subject_rows: list[dict[str, Any]] = []

    pred_dir = output_dir / "nifti_predictions"
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        subject_id = str(row["subject_id"])
        record_id = str(row["record_id"])

        img_file = data_root / row["image_path"]
        nii_img = nib.load(str(img_file))
        raw_img = np.asanyarray(nii_img.dataobj)
        zooms = nii_img.header.get_zooms()

        has_label = bool(row.get("has_label", False)) and pd.notna(row.get("label_path"))
        raw_label = None
        if has_label:
            lbl_file = data_root / row["label_path"]
            if lbl_file.exists():
                raw_label = np.rint(np.asanyarray(nib.load(str(lbl_file)).dataobj)).astype(np.int16)

        # -------------------------------------------------------------
        # CASE 1: 3D SAX
        # -------------------------------------------------------------
        if view == "SAX":
            if raw_img.ndim == 4:
                raw_img = np.squeeze(raw_img)
            if raw_label is not None and raw_label.ndim == 4:
                raw_label = np.squeeze(raw_label)

            orig_spacing = tuple(float(v) for v in zooms[:3])
            proc_img, transform = preprocess_spatial(
                raw_img,
                source_spacing=orig_spacing,
                target_spacing=target_spacing,
                target_shape=target_shape,
                interpolation_order=1,
                intensity_percentiles=percentiles,
            )

            proc_dhw = np.transpose(proc_img, (2, 0, 1))  # (16, 192, 192)
            tensor = torch.from_numpy(proc_dhw[None, None, ...]).to(device, dtype=torch.float32)
            with torch.no_grad():
                logits = model(tensor)  # (1, num_classes, 16, 192, 192)
                pred_dhw = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int16)
            pred_processed = np.transpose(pred_dhw, (1, 2, 0))  # back to (192, 192, 16)

            restored_pred = invert_spatial_mask(pred_processed, transform)

        # -------------------------------------------------------------
        # CASE 2: 2D LAX (2CH, 4CH, RAS)
        # -------------------------------------------------------------
        else:
            orig_spacing = tuple(float(v) for v in zooms[:2])
            d_count = raw_img.shape[2] if raw_img.ndim >= 3 else 1
            restored_slices = []

            for s in range(d_count):
                slice_img = raw_img[:, :, s] if raw_img.ndim >= 3 else raw_img
                p_sl, trans_s = preprocess_spatial(
                    slice_img,
                    source_spacing=orig_spacing,
                    target_spacing=target_spacing,
                    target_shape=target_shape,
                    interpolation_order=1,
                    intensity_percentiles=percentiles,
                )
                t_sl = torch.from_numpy(p_sl[None, None, ...]).to(device, dtype=torch.float32)
                with torch.no_grad():
                    l_sl = model(t_sl)
                    p_out = torch.argmax(l_sl, dim=1)[0].cpu().numpy().astype(np.int16)
                rest_s = invert_spatial_mask(p_out, trans_s)
                restored_slices.append(rest_s)

            restored_pred = np.stack(restored_slices, axis=-1) if raw_img.ndim >= 3 else restored_slices[0]

        # Save restored NIfTI prediction
        if save_predictions:
            out_nii = nib.Nifti1Image(restored_pred.astype(np.uint8), nii_img.affine, nii_img.header)
            nib.save(out_nii, str(pred_dir / f"{record_id}_pred.nii.gz"))

        # Calculate metrics if ground truth is available
        row_metrics: dict[str, Any] = {
            "record_id": record_id,
            "subject_id": subject_id,
            "view": view,
        }

        if raw_label is not None:
            # Overlap metrics
            dice_dict = dice_score(restored_pred, raw_label, num_classes=num_classes)
            iou_dict = iou_score(restored_pred, raw_label, num_classes=num_classes)

            for c in range(1, num_classes):
                c_name = label_names.get(c, f"class_{c}")
                row_metrics[f"dice_{c_name}"] = dice_dict.get(c, float("nan"))
                row_metrics[f"iou_{c_name}"] = iou_dict.get(c, float("nan"))
                # HD95
                full_spacing = tuple(float(v) for v in zooms[:restored_pred.ndim])
                hd95_val = hd95_binary(restored_pred == c, raw_label == c, spacing=full_spacing)
                row_metrics[f"hd95_{c_name}_mm"] = hd95_val if hd95_val is not None else float("nan")

            # Clinical scar metrics (available for SAX, 2CH, 4CH)
            if 3 in label_names and label_names[3] == "scar":
                full_spacing = tuple(float(v) for v in zooms[:restored_pred.ndim])
                scar_pred = calculate_scar_metrics(restored_pred, spacing=full_spacing)
                scar_true = calculate_scar_metrics(raw_label, spacing=full_spacing)

                row_metrics["pred_scar_volume_ml"] = scar_pred.scar_volume_ml
                row_metrics["true_scar_volume_ml"] = scar_true.scar_volume_ml
                row_metrics["scar_volume_error_ml"] = abs(scar_pred.scar_volume_ml - scar_true.scar_volume_ml)

                row_metrics["pred_scar_mass_g"] = scar_pred.scar_mass_g
                row_metrics["true_scar_mass_g"] = scar_true.scar_mass_g
                row_metrics["scar_mass_error_g"] = abs(scar_pred.scar_mass_g - scar_true.scar_mass_g)

                row_metrics["pred_scar_fraction"] = scar_pred.scar_fraction_of_myo_plus_scar
                row_metrics["true_scar_fraction"] = scar_true.scar_fraction_of_myo_plus_scar

        per_subject_rows.append(row_metrics)

    subject_df = pd.DataFrame(per_subject_rows)
    subject_df.to_csv(output_dir / "per_subject_metrics.csv", index=False)

    # Aggregate summary
    summary_rows = []
    metric_cols = [c for c in subject_df.columns if c.startswith(("dice_", "iou_", "hd95_", "scar_"))]
    for col in metric_cols:
        series = subject_df[col].dropna()
        series_clean = series[np.isfinite(series)]
        summary_rows.append({
            "metric": col,
            "mean": float(series_clean.mean()) if len(series_clean) else float("nan"),
            "std": float(series_clean.std()) if len(series_clean) else float("nan"),
            "median": float(series_clean.median()) if len(series_clean) else float("nan"),
            "count": int(len(series_clean)),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "per_class_summary.csv", index=False)

    return subject_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LGE model and compute scar metrics.")
    parser.add_argument("--config", required=True, help="Path to config snapshot YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--output-dir", default=None, help="Directory to save evaluation reports")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(config.get("num_classes", 5))

    # Model
    model = build_model(config["model_name"], **{**config.get("model", {}), "num_classes": num_classes})
    ckpt_path = ROOT / args.checkpoint
    if not ckpt_path.exists():
        candidates = [
            ckpt_path.parent / "best_scar_dice.pt",
            ckpt_path.parent / "best_mean_dice.pt",
            ckpt_path.parent / "best_loss.pt",
            ckpt_path.parent / "last.pt",
        ]
        found = False
        for cand in candidates:
            if cand.exists():
                logger.warning("Checkpoint '%s' not found, falling back to '%s'", ckpt_path, cand)
                ckpt_path = cand
                found = True
                break
        if not found:
            raise FileNotFoundError(f"No valid checkpoint found in {ckpt_path.parent}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    logger.info("Loaded model from %s (epoch %d)", ckpt_path, ckpt.get("epoch", -1))

    # Data
    splits_dir = ROOT / config["data"]["splits_dir"]
    df = pd.read_csv(splits_dir / f"{args.split}.csv")
    view = str(config.get("view", "SAX")).upper()
    df = df[df["view"] == view]
    data_root = ROOT / config["data"]["raw_root"]

    run_dir = config_path.parent
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "evaluation" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluating %d records on '%s' split for view '%s'...", len(df), args.split, view)
    subject_df, summary_df = evaluate_split(
        model=model,
        df=df,
        data_root=data_root,
        config=config,
        device=device,
        save_predictions=True,
        output_dir=output_dir,
    )

    logger.info("Evaluation complete! Results exported to:\n  - %s\n  - %s",
                output_dir / "per_subject_metrics.csv",
                output_dir / "per_class_summary.csv")
    print("\n--- Summary Performance ---")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
