"""Trainer class for LGE Cardiac MRI segmentation models."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.metrics import dice_score
from training.postprocess import decode_with_rules, enforce_anatomical_constraints

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


class Trainer:
    """Multi-view LGE Segmentation Trainer."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        scheduler: _LRScheduler | None = None,
        device: torch.device,
        num_classes: int | None = None,
        checkpoint_dir: Path | str | None = None,
        run_dir: Path | str | None = None,
        config: dict | None = None,
        view: str | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}

        # Resolve num_classes
        if num_classes is not None:
            self.num_classes = int(num_classes)
        elif "num_classes" in self.config:
            self.num_classes = int(self.config["num_classes"])
        else:
            self.num_classes = 5

        # Resolve view
        if view is not None:
            self.view = str(view).upper()
        elif "view" in self.config:
            self.view = str(self.config["view"]).upper()
        else:
            self.view = "SAX"

        # Resolve run_dir & checkpoint_dir & logs_dir
        if run_dir is not None:
            self.run_dir = Path(run_dir)
        elif checkpoint_dir is not None:
            self.run_dir = Path(checkpoint_dir).parent
        else:
            self.run_dir = Path("outputs/runs")

        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = self.run_dir / "checkpoints"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

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
            max_grad_norm = float(self.config.get("training", {}).get("max_grad_norm", 1.0))
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
            self.optimizer.step()

            losses.append(float(loss.detach().cpu()))

        if self.scheduler is not None:
            self.scheduler.step()

        return float(np.mean(losses)) if losses else math.nan

    def _validate_epoch(self, dataloader: DataLoader) -> tuple[float, dict[str, float]]:
        """Validate model on the validation DataLoader in preprocessed batch space.
        
        Uses Subject-Level Macro-Dice aligned with Metrics Reloaded (Nature Methods 2024)
        and training/evaluate.py standards. Slices belonging to the same subject are aggregated
        before computing per-subject Dice, and True Negative subjects (GT=0, Pred=0) evaluate to 1.0.
        """
        self.model.eval()
        losses = []

        # Subject-level voxel counters: subject_id -> class_id -> count
        subj_inter: dict[str, dict[int, int]] = {}
        subj_gt: dict[str, dict[int, int]] = {}
        subj_pred: dict[str, dict[int, int]] = {}
        sample_idx = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device, dtype=torch.float32)
                labels = batch["label"].to(self.device, dtype=torch.long)

                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                losses.append(float(loss.detach().cpu()))

                # Decode logits: One-vs-Rest or Softmax Argmax
                post_cfg = self.config.get("postprocess", {})
                use_rules = post_cfg.get("use_rules", True)

                if logits.shape[1] == self.num_classes - 1:
                    preds_tensor = decode_with_rules(logits, view=self.view)
                elif use_rules and logits.shape[1] == self.num_classes:
                    preds_tensor = decode_with_rules(logits[:, 1:], view=self.view)
                else:
                    preds_tensor = torch.argmax(logits, dim=1)

                preds_np = preds_tensor.cpu().numpy()

                # Apply anatomical constraints if enabled
                if post_cfg.get("anatomical_constraint", True):
                    cleaned_preds = []
                    for b in range(preds_np.shape[0]):
                        cleaned = enforce_anatomical_constraints(
                            preds_np[b],
                            scar_class=3,
                            myo_class=2,
                            dilation_voxels=int(post_cfg.get("dilation_voxels", 1)),
                            tolerance_mm=float(post_cfg.get("tolerance_mm", 2.5)),
                            min_scar_voxels=int(post_cfg.get("min_scar_voxels", 5)),
                            min_scar_volume_mm3=float(post_cfg.get("min_scar_volume_mm3", 15.0)),
                        )
                        cleaned_preds.append(cleaned)
                    preds = np.stack(cleaned_preds, axis=0)
                else:
                    preds = preds_np

                targets = labels.cpu().numpy()

                # Resolve subject identifiers from batch metadata or fallback to sample index
                batch_subjects = batch.get("subject_id")
                if batch_subjects is None:
                    batch_subjects = batch.get("record_id")

                for b in range(len(preds)):
                    if batch_subjects is not None:
                        s_id = str(batch_subjects[b]) if not isinstance(batch_subjects, str) else str(batch_subjects)
                    else:
                        s_id = f"sample_{sample_idx}"
                    sample_idx += 1

                    if s_id not in subj_inter:
                        subj_inter[s_id] = {c: 0 for c in range(1, self.num_classes)}
                        subj_gt[s_id] = {c: 0 for c in range(1, self.num_classes)}
                        subj_pred[s_id] = {c: 0 for c in range(1, self.num_classes)}

                    for c in range(1, self.num_classes):
                        p_mask = preds[b] == c
                        t_mask = targets[b] == c
                        subj_inter[s_id][c] += int((p_mask & t_mask).sum())
                        subj_gt[s_id][c] += int(t_mask.sum())
                        subj_pred[s_id][c] += int(p_mask.sum())

        val_loss = float(np.mean(losses)) if losses else math.nan
        metrics_summary: dict[str, float] = {}
        all_subjects = list(subj_inter.keys())

        for c in range(1, self.num_classes):
            c_subject_dices: list[float] = []
            for s_id in all_subjects:
                gt_c = subj_gt[s_id][c]
                pred_c = subj_pred[s_id][c]
                inter_c = subj_inter[s_id][c]

                # Metrics Reloaded (2024) Subject-Level Formulation
                if gt_c == 0 and pred_c == 0:
                    dice_val = 1.0  # Symmetric True Negative
                elif gt_c == 0 or pred_c == 0:
                    dice_val = 0.0  # Complete False Positive or False Negative
                else:
                    dice_val = float(2.0 * inter_c / (gt_c + pred_c))
                c_subject_dices.append(dice_val)

            metrics_summary[f"dice_class_{c}"] = float(np.mean(c_subject_dices)) if c_subject_dices else float("nan")

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
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "val_loss": val_loss,
                "metrics": metrics,
            },
            path,
        )

    def load_checkpoint(self, path: Path | str) -> dict[str, Any]:
        """Load checkpoint state into model, optimizer, and scheduler."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt

    def _save_history(self) -> None:
        """Save history to JSON and CSV in both run_dir and run_dir/logs."""
        try:
            # 1. Save to run_dir / logs / training_history.json
            log_json = self.logs_dir / "training_history.json"
            log_json.write_text(json.dumps(self.history, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

            # 2. Save to run_dir / training_history.json
            root_json = self.run_dir / "training_history.json"
            root_json.write_text(json.dumps(self.history, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

            # 3. Save CSV version
            if self.history:
                df = pd.DataFrame(self.history)
                df.to_csv(self.logs_dir / "training_history.csv", index=False)
                df.to_csv(self.run_dir / "training_history.csv", index=False)
        except Exception as e:
            logger.warning("Could not save training history: %s", e)

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

            # 6. If view has scar, ensure best_scar_dice.pt is initialized
            if self.view in ("SAX", "2CH", "4CH") and not (self.checkpoint_dir / "best_scar_dice.pt").exists():
                self.save_checkpoint("best_scar_dice", epoch, val_loss, val_metrics)

            # Save training history per epoch
            self._save_history()

            if stopper is not None:
                eval_val = val_loss if monitor_metric == "val_loss" else val_metrics.get(monitor_metric, float("nan"))
                if stopper.step(eval_val):
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break

        self._save_history()
        return self.history


# Alias for backwards compatibility
LgeTrainer = Trainer
