"""Evaluation and clinical scar quantification for Zhang (2021) Cascaded 2D-3D model.

Usage:
    python training/evaluate_zhang.py \
        --checkpoint outputs/runs/zhang_cascaded_exp01/checkpoints/best.pt \
        --config training/config/models/zhang_cascaded.yaml \
        --split validation \
        --output-dir outputs/reports/zhang_cascaded_eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

# Make repo root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.dataset.lge_dataset import LgeSaxDataset
from training.metrics import (
    calculate_scar_metrics,
    dice_score,
    hd95_binary,
    iou_score,
)
from training.models.zhang_cascaded_unet import ZhangCascadedUNet
from training.postprocessing.zhang_postprocess import zhang_postprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ZhangEvaluator")


def evaluate_model(
    model: ZhangCascadedUNet,
    dataset: LgeSaxDataset,
    device: torch.device,
    num_classes: int = 5,
    postprocess_cfg: Optional[Dict[str, Any]] = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 10.0),
) -> Dict[str, Any]:
    """Run evaluation and compute clinical metrics over all cases in dataset."""
    model.eval()
    pp_cfg = postprocess_cfg or {}
    apply_pp = bool(pp_cfg.get("enabled", True))
    min_size = int(pp_cfg.get("min_size_voxels", 8))
    enforce_proximity = bool(pp_cfg.get("enforce_myo_proximity", True))

    records = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device, dtype=torch.float32)  # (1, 1, D, H, W)
            target = sample["label"].cpu().numpy()  # (D, H, W)
            case_id = sample.get("case_id", f"case_{idx:03d}")

            outputs = model.forward_stages(image)
            raw_pred = torch.argmax(outputs["fine_logits"], dim=1).squeeze(0).cpu().numpy()
            coarse_pred = torch.argmax(outputs["coarse_logits"], dim=1).squeeze(0).cpu().numpy()

            if apply_pp:
                final_pred = zhang_postprocess(
                    raw_pred,
                    min_size_voxels=min_size,
                    enforce_myo_proximity=enforce_proximity,
                )
            else:
                final_pred = raw_pred

            # Metrics
            d_fine = dice_score(final_pred, target, num_classes=num_classes)
            d_coarse = dice_score(coarse_pred, target, num_classes=num_classes)
            iou_fine = iou_score(final_pred, target, num_classes=num_classes)

            # HD95 for Myo (class 2) and Scar (class 3)
            hd95_myo = hd95_binary(final_pred == 2, target == 2, spacing=spacing)
            hd95_scar = hd95_binary(final_pred == 3, target == 3, spacing=spacing)

            # Scar & Myocardium volumes
            pred_quant = calculate_scar_metrics(final_pred, spacing=spacing)
            true_quant = calculate_scar_metrics(target, spacing=spacing)

            scar_vol_pred_ml = float(pred_quant.scar_volume_ml)
            scar_vol_true_ml = float(true_quant.scar_volume_ml)
            scar_vol_diff_ml = abs(scar_vol_pred_ml - scar_vol_true_ml)

            # Percentage of Infarcted Myocardium (PIM %)
            myo_vol_pred_ml = float(np.sum(final_pred == 2) * np.prod(spacing) / 1000.0)
            myo_vol_true_ml = float(np.sum(target == 2) * np.prod(spacing) / 1000.0)

            total_myo_pred = myo_vol_pred_ml + scar_vol_pred_ml
            total_myo_true = myo_vol_true_ml + scar_vol_true_ml

            pim_pred = (scar_vol_pred_ml / total_myo_pred * 100.0) if total_myo_pred > 0 else 0.0
            pim_true = (scar_vol_true_ml / total_myo_true * 100.0) if total_myo_true > 0 else 0.0
            pim_diff = abs(pim_pred - pim_true)

            case_row = {
                "case_id": case_id,
                "dice_myo": d_fine.get(2, float("nan")),
                "dice_scar": d_fine.get(3, float("nan")),
                "dice_coarse_scar": d_coarse.get(3, float("nan")),
                "iou_scar": iou_fine.get(3, float("nan")),
                "hd95_myo_mm": hd95_myo,
                "hd95_scar_mm": hd95_scar,
                "scar_volume_pred_ml": scar_vol_pred_ml,
                "scar_volume_true_ml": scar_vol_true_ml,
                "scar_volume_diff_ml": scar_vol_diff_ml,
                "pim_pred_pct": pim_pred,
                "pim_true_pct": pim_true,
                "pim_diff_pct": pim_diff,
            }
            records.append(case_row)

    df = pd.DataFrame(records)

    def mean_std(series: pd.Series) -> str:
        s = series.dropna()
        return f"{s.mean():.4f} ± {s.std():.4f}" if len(s) > 0 else "N/A"

    summary = {
        "num_cases": len(df),
        "mean_dice_myo": float(df["dice_myo"].dropna().mean()),
        "mean_dice_scar": float(df["dice_scar"].dropna().mean()),
        "mean_coarse_dice_scar": float(df["dice_coarse_scar"].dropna().mean()),
        "mean_scar_vol_diff_ml": float(df["scar_volume_diff_ml"].dropna().mean()),
        "mean_pim_diff_pct": float(df["pim_diff_pct"].dropna().mean()),
        "mean_hd95_scar_mm": float(df["hd95_scar_mm"].dropna().mean()) if len(df["hd95_scar_mm"].dropna()) > 0 else None,
        "details_summary": {
            "myo_dice": mean_std(df["dice_myo"]),
            "scar_dice": mean_std(df["dice_scar"]),
            "scar_vol_diff_ml": mean_std(df["scar_volume_diff_ml"]),
            "pim_diff_pct": mean_std(df["pim_diff_pct"]),
        },
    }

    return {"summary": summary, "per_case": df}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Zhang (2021) Cascaded 2D-3D Model.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--config", default="training/config/models/zhang_cascaded.yaml")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--output-dir", default="outputs/reports/zhang_cascaded_eval")
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(config.get("num_classes", 5))

    # Initialize model
    model_cfg = config.get("model", {})
    model = ZhangCascadedUNet(
        in_channels=int(model_cfg.get("in_channels", 1)),
        num_classes=num_classes,
        coarse_features=model_cfg.get("coarse_features", [32, 64, 128, 256]),
        fine_features=model_cfg.get("fine_features", [32, 64, 128, 256]),
        dropout=0.0,
    ).to(device)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    logger.info("Loaded checkpoint from %s (Epoch %s)", ckpt_path, ckpt.get("epoch", "N/A"))

    # Load dataset
    splits_dir = ROOT / config.get("data", {}).get("splits_dir", "data/processed/splits")
    split_file = splits_dir / f"{args.split}.csv"
    if not split_file.exists():
        logger.error("Split file %s does not exist.", split_file)
        return

    df_split = pd.read_csv(split_file)
    df_split = df_split[df_split["view"] == "SAX"]

    pp = config.get("preprocessing", {})
    target_shape = tuple(pp.get("target_shape", [192, 192, 16]))
    target_spacing = tuple(pp.get("target_spacing", [1.0, 1.0, 10.0]))
    percentiles = tuple(pp.get("intensity_percentiles", [0.95, 99.5]))

    cache_dir = ROOT / config.get("data", {}).get("cache_dir", "data/processed/cache")
    raw_root = ROOT / config.get("data", {}).get("raw_root", "data/LGE_MULTI")

    dataset = LgeSaxDataset(
        records=df_split,
        data_root=raw_root,
        cache_dir=cache_dir if cache_dir.exists() else None,
        target_shape=target_shape,
        target_spacing=target_spacing,
        intensity_percentiles=percentiles,
        augment=False,
    )

    results = evaluate_model(
        model=model,
        dataset=dataset,
        device=device,
        num_classes=num_classes,
        postprocess_cfg=config.get("postprocessing", {}),
        spacing=target_spacing,
    )

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_file = out_dir / "summary.json"
    summary_file.write_text(json.dumps(results["summary"], indent=2), encoding="utf-8")

    per_case_csv = out_dir / "per_case_metrics.csv"
    results["per_case"].to_csv(per_case_csv, index=False)

    logger.info("Evaluation Complete! Summary:\n%s", json.dumps(results["summary"]["details_summary"], indent=2))
    logger.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
