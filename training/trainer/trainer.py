"""Trainer class for LGE Cardiac MRI segmentation models."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.metrics import dice_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-4, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float = -math.inf if mode == "max" else math.inf
        self.counter = 0
        self.early_stop = False

    def step(self, score: float) -> bool:
        if np.isnan(score):
            return self.early_stop

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class LgeTrainer:
    """Multi-view LGE Segmentation Trainer."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        scheduler: _LRScheduler | None = None,
        device: torch.device,
        num_classes: int,
        checkpoint_dir: Path,
        view: str = "SAX",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.view = view
        self.history: list[dict[str, Any]] = []

    def _train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        losses = []

        for batch in dataloader:
            images = batch["image"].to(self.device, dtype=torch.float32)
            labels = batch["label"].to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)

            loss.backward()
            self.optimizer.step()

            losses.append(float(loss.detach().cpu()))

        if self.scheduler is not None:
            self.scheduler.step()

        return float(np.mean(losses)) if losses else math.nan

    def _validate_epoch(self, dataloader: DataLoader) -> tuple[float, dict[str, float]]:
        self.model.eval()
        losses = []
        all_dices: list[dict[int, float]] = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device, dtype=torch.float32)
                labels = batch["label"].to(self.device, dtype=torch.long)

                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                losses.append(float(loss.detach().cpu()))

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                targets = labels.cpu().numpy()

                for b in range(len(preds)):
                    d_dict = dice_score(preds[b], targets[b], num_classes=self.num_classes)
                    all_dices.append(d_dict)

        val_loss = float(np.mean(losses)) if losses else math.nan
        metrics_summary: dict[str, float] = {}

        for c in range(1, self.num_classes):
            c_scores = [d[c] for d in all_dices if c in d and not np.isnan(d[c])]
            metrics_summary[f"dice_class_{c}"] = float(np.mean(c_scores)) if c_scores else float("nan")

        # Class 3 is SCAR in LGE SAX, 2CH, 4CH
        if "dice_class_3" in metrics_summary:
            metrics_summary["dice_scar"] = metrics_summary["dice_class_3"]

        valid_means = [v for k, v in metrics_summary.items() if k.startswith("dice_class_") and not np.isnan(v)]
        metrics_summary["mean_dice"] = float(np.mean(valid_means)) if valid_means else float("nan")

        return val_loss, metrics_summary

    def save_checkpoint(self, name: str, epoch: int, val_loss: float, metrics: dict[str, float]) -> None:
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
        early_stopping_cfg: dict | None = None,
    ) -> list[dict[str, Any]]:
        es_cfg = early_stopping_cfg or {}
        monitor_metric = es_cfg.get("monitor", "val_loss")
        monitor_mode = es_cfg.get("mode", "min" if "loss" in monitor_metric else "max")

        stopper = (
            EarlyStopping(
                patience=int(es_cfg.get("patience", 30)),
                min_delta=float(es_cfg.get("min_delta", 1e-4)),
                mode=monitor_mode,
            )
            if es_cfg.get("enabled", True)
            else None
        )

        best_scar_dice = -1.0
        best_mean_dice = -1.0
        best_val_loss = math.inf

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate_epoch(val_loader)

            scar_dice_str = f"  dice_scar={val_metrics.get('dice_scar', float('nan')):.4f}" if "dice_scar" in val_metrics else ""
            mean_dice_str = f"  mean_dice={val_metrics.get('mean_dice', float('nan')):.4f}" if "mean_dice" in val_metrics else ""

            logger.info(
                "Epoch %3d/%3d — train_loss=%.4f  val_loss=%.4f%s%s",
                epoch,
                epochs,
                train_loss,
                val_loss,
                scar_dice_str,
                mean_dice_str,
            )

            row: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics,
            }
            self.history.append(row)

            # 1. Track best loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best_loss", epoch, val_loss, val_metrics)

            # 2. Track best model for views with Scar (SAX, 2CH, 4CH)
            has_scar = "dice_scar" in val_metrics and not np.isnan(val_metrics["dice_scar"])
            if has_scar:
                current_scar_dice = val_metrics["dice_scar"]
                if current_scar_dice > best_scar_dice:
                    best_scar_dice = current_scar_dice
                    self.save_checkpoint("best_scar_dice", epoch, val_loss, val_metrics)
                    self.save_checkpoint("best", epoch, val_loss, val_metrics)

            # 3. Track best model for views without Scar (e.g. RAS) using mean_dice
            current_mean_dice = val_metrics.get("mean_dice", -1.0)
            if not has_scar and not np.isnan(current_mean_dice) and current_mean_dice > best_mean_dice:
                best_mean_dice = current_mean_dice
                self.save_checkpoint("best_mean_dice", epoch, val_loss, val_metrics)
                self.save_checkpoint("best", epoch, val_loss, val_metrics)

            # 4. Always save latest checkpoint
            self.save_checkpoint("last", epoch, val_loss, val_metrics)

            # 5. If best.pt still doesn't exist (e.g. epoch 1 all dices were 0), save best.pt from best_loss
            if not (self.checkpoint_dir / "best.pt").exists():
                self.save_checkpoint("best", epoch, val_loss, val_metrics)

            if stopper is not None:
                eval_val = val_loss if monitor_metric == "val_loss" else val_metrics.get(monitor_metric, float("nan"))
                if stopper.step(eval_val):
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break

        return self.history
