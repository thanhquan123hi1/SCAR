"""Dedicated training pipeline for Zhang (2021) Cascaded 2D-3D Architecture on LGE Cardiac MRI.

Usage:
    python training/train_zhang.py \
        --config training/config/models/zhang_cascaded.yaml \
        --run-id zhang_cascaded_exp01
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Make repo root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.dataset.lge_dataset import LgeSaxDataset
from training.loss.zhang_combo_loss import ZhangCascadedLoss, ZhangComboLoss
from training.metrics import dice_score
from training.models.zhang_cascaded_unet import ZhangCascadedUNet
from training.postprocessing.zhang_postprocess import zhang_postprocess
from training.trainer.trainer import EarlyStopping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ZhangTrainer")


def set_seed(seed: int) -> None:
    """Ensure 100% deterministic reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def merge_configs(base_path: Path, model_path: Path) -> dict:
    """Load base config then override with model-specific config."""
    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    model_cfg = yaml.safe_load(model_path.read_text(encoding="utf-8")) if model_path.exists() else {}

    def deep_merge(base_d: dict, override_d: dict) -> dict:
        result = dict(base_d)
        for key, value in override_d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    return deep_merge(base, model_cfg)


class ZhangCascadedTrainer:
    """Trainer specifically tailored for Zhang (2021) Cascaded 2D-3D U-Net."""

    def __init__(
        self,
        *,
        model: ZhangCascadedUNet,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device,
        num_classes: int,
        checkpoint_dir: Path,
        postprocess_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.postprocess_cfg = postprocess_cfg or {}
        self.history: List[Dict[str, Any]] = []

    def _train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        losses = []

        for batch in dataloader:
            images = batch["image"].to(self.device, dtype=torch.float32)
            raw_labels = batch["label"].to(self.device, dtype=torch.long)
            # Ensure label indices are strictly inside valid class range [0, num_classes-1]
            labels = torch.clamp(raw_labels, 0, self.num_classes - 1)

            self.optimizer.zero_grad()
            outputs = self.model.forward_stages(images)
            loss = self.loss_fn(outputs, labels)

            loss.backward()
            self.optimizer.step()

            losses.append(float(loss.detach().cpu()))

        if self.scheduler is not None:
            self.scheduler.step()

        return float(np.mean(losses)) if losses else math.nan

    def _validate_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        losses = []
        all_dices: List[Dict[int, float]] = []
        all_dices_coarse: List[Dict[int, float]] = []

        apply_pp = bool(self.postprocess_cfg.get("enabled", True))
        min_size = int(self.postprocess_cfg.get("min_size_voxels", 8))
        enforce_proximity = bool(self.postprocess_cfg.get("enforce_myo_proximity", True))

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device, dtype=torch.float32)
                raw_labels = batch["label"].to(self.device, dtype=torch.long)
                labels = torch.clamp(raw_labels, 0, self.num_classes - 1)

                outputs = self.model.forward_stages(images)
                loss = self.loss_fn(outputs, labels)
                losses.append(float(loss.detach().cpu()))

                preds = torch.argmax(outputs["fine_logits"], dim=1).cpu().numpy()
                preds_coarse = torch.argmax(outputs["coarse_logits"], dim=1).cpu().numpy()
                targets = labels.cpu().numpy()

                for b in range(len(preds)):
                    pred_b = preds[b]
                    if apply_pp:
                        pred_b = zhang_postprocess(
                            pred_b,
                            min_size_voxels=min_size,
                            enforce_myo_proximity=enforce_proximity,
                        )

                    d_dict = dice_score(pred_b, targets[b], num_classes=self.num_classes)
                    d_coarse = dice_score(preds_coarse[b], targets[b], num_classes=self.num_classes)
                    all_dices.append(d_dict)
                    all_dices_coarse.append(d_coarse)

        val_loss = float(np.mean(losses)) if losses else math.nan
        metrics_summary: Dict[str, float] = {}

        for c in range(1, self.num_classes):
            c_scores = [d[c] for d in all_dices if c in d and not np.isnan(d[c])]
            metrics_summary[f"dice_class_{c}"] = float(np.mean(c_scores)) if c_scores else float("nan")

            c_coarse = [d[c] for d in all_dices_coarse if c in d and not np.isnan(d[c])]
            metrics_summary[f"coarse_dice_class_{c}"] = float(np.mean(c_coarse)) if c_coarse else float("nan")

        # Class 3 is Scar
        if "dice_class_3" in metrics_summary:
            metrics_summary["dice_scar"] = metrics_summary["dice_class_3"]
        if "coarse_dice_class_3" in metrics_summary:
            metrics_summary["coarse_dice_scar"] = metrics_summary["coarse_dice_class_3"]

        valid_means = [v for k, v in metrics_summary.items() if k.startswith("dice_class_") and not np.isnan(v)]
        metrics_summary["mean_dice"] = float(np.mean(valid_means)) if valid_means else float("nan")

        return val_loss, metrics_summary

    def save_checkpoint(self, name: str, epoch: int, val_loss: float, metrics: dict) -> None:
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "metrics": metrics,
            },
            path,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        epochs: int,
        early_stopping_cfg: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        es_cfg = early_stopping_cfg or {}
        monitor_metric = es_cfg.get("monitor", "dice_scar")
        monitor_mode = es_cfg.get("mode", "max")

        stopper = (
            EarlyStopping(
                patience=int(es_cfg.get("patience", 20)),
                min_delta=float(es_cfg.get("min_delta", 1e-4)),
                mode=monitor_mode,
            )
            if es_cfg.get("enabled", True)
            else None
        )

        best_scar_dice = -1.0
        best_val_loss = math.inf

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate_epoch(val_loader)

            scar_dice = val_metrics.get("dice_scar", float("nan"))
            coarse_scar = val_metrics.get("coarse_dice_scar", float("nan"))
            mean_dice = val_metrics.get("mean_dice", float("nan"))

            logger.info(
                "Epoch %3d/%3d — train_loss=%.4f  val_loss=%.4f  fine_scar_dice=%.4f (coarse=%.4f)  mean_dice=%.4f",
                epoch,
                epochs,
                train_loss,
                val_loss,
                scar_dice,
                coarse_scar,
                mean_dice,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics,
            }
            self.history.append(row)

            # Checkpoints
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best_loss", epoch, val_loss, val_metrics)

            if not np.isnan(scar_dice) and scar_dice > best_scar_dice:
                best_scar_dice = scar_dice
                self.save_checkpoint("best_scar_dice", epoch, val_loss, val_metrics)
                self.save_checkpoint("best", epoch, val_loss, val_metrics)

            self.save_checkpoint("last", epoch, val_loss, val_metrics)

            if not (self.checkpoint_dir / "best.pt").exists():
                self.save_checkpoint("best", epoch, val_loss, val_metrics)

            if stopper is not None:
                eval_val = val_loss if monitor_metric == "val_loss" else val_metrics.get(monitor_metric, float("nan"))
                if stopper.step(eval_val):
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break

        return self.history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Zhang Cascaded 2D-3D Segmentation Model.")
    parser.add_argument("--config", default="training/config/models/zhang_cascaded.yaml", help="Path to config YAML")
    parser.add_argument("--run-id", default="zhang_cascaded_run01", help="Unique run identifier")
    parser.add_argument("--base-config", default="training/config/base.yaml")
    args = parser.parse_args()

    config = merge_configs(ROOT / args.base_config, ROOT / args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    logger.info("Deterministic seed set to: %d", seed)

    req_device = str(config.get("device", "auto")).lower()
    device = torch.device("cuda" if (req_device == "auto" and torch.cuda.is_available()) else ("cpu" if req_device == "auto" else req_device))
    logger.info("Using device: %s", device)

    run_dir = ROOT / config.get("outputs", {}).get("run_root", "outputs/runs") / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config_snapshot.yaml").write_text(
        yaml.dump(config, allow_unicode=True), encoding="utf-8"
    )

    # Load splits
    splits_dir = ROOT / config["data"]["splits_dir"]
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "validation.csv")

    train_df = train_df[train_df["view"] == "SAX"]
    val_df = val_df[val_df["view"] == "SAX"]
    logger.info("Dataset filtered: Train=%d, Val=%d (View=SAX)", len(train_df), len(val_df))

    pp = config.get("preprocessing", {})
    target_shape = tuple(pp["target_shape"])
    target_spacing = tuple(pp["target_spacing"])
    percentiles = tuple(pp["intensity_percentiles"]) if pp.get("intensity_percentiles") else None

    cache_dir = ROOT / config["data"].get("cache_dir", "data/processed/cache")
    raw_root = ROOT / config["data"]["raw_root"]

    train_ds = LgeSaxDataset(
        records=train_df,
        data_root=raw_root,
        cache_dir=cache_dir if cache_dir.exists() else None,
        target_shape=target_shape,
        target_spacing=target_spacing,
        intensity_percentiles=percentiles,
        augment=bool(config.get("training", {}).get("augment", True)),
    )
    val_ds = LgeSaxDataset(
        records=val_df,
        data_root=raw_root,
        cache_dir=cache_dir if cache_dir.exists() else None,
        target_shape=target_shape,
        target_spacing=target_spacing,
        intensity_percentiles=percentiles,
        augment=False,
    )

    batch_size = int(config.get("training", {}).get("batch_size", 2))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=bool(config["data"].get("pin_memory", False)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
    )

    num_classes = int(config.get("num_classes", 5))
    model_cfg = config.get("model", {})
    model = ZhangCascadedUNet(
        in_channels=int(model_cfg.get("in_channels", 1)),
        num_classes=num_classes,
        coarse_features=model_cfg.get("coarse_features", [32, 64, 128, 256]),
        fine_features=model_cfg.get("fine_features", [32, 64, 128, 256]),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_one_hot=bool(model_cfg.get("use_one_hot", False)),
    ).to(device)

    # Optimizer & Scheduler
    lr = float(config.get("training", {}).get("learning_rate", 1e-4))
    wd = float(config.get("training", {}).get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(config.get("training", {}).get("epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Loss
    loss_cfg = config.get("loss", {})
    class_weights = loss_cfg.get("class_weights", [0.5, 1.0, 1.5, 3.0, 1.0])
    loss_fn = ZhangCascadedLoss(
        num_classes=num_classes,
        coarse_loss_weight=float(loss_cfg.get("coarse_loss_weight", 0.5)),
        ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        class_weights=class_weights,
    ).to(device)

    # Trainer
    trainer = ZhangCascadedTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        num_classes=num_classes,
        checkpoint_dir=run_dir / "checkpoints",
        postprocess_cfg=config.get("postprocessing", {}),
    )

    logger.info("Starting Zhang (2021) Cascaded training for %d epochs...", epochs)
    trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        early_stopping_cfg=config.get("training", {}).get("early_stopping"),
    )
    logger.info("Training finished. Checkpoints saved to %s", run_dir / "checkpoints")


if __name__ == "__main__":
    main()
