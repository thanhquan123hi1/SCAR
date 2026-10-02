"""Segmentation loss functions."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    """Per-image equally weighted class Dice with squared denominator.

    Includes background; reductions are float32 even under autocast. Per-image
    reduction makes sample-weighted gradient accumulation well-defined.
    """

    def __init__(self, n_classes=4, smooth=1e-5):
        super().__init__()
        if not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError("n_classes must be an integer >= 2")
        if not math.isfinite(smooth) or smooth <= 0:
            raise ValueError("Dice smooth must be finite and positive")
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, inputs, target, weight=None, softmax=False):
        probabilities = inputs.float().softmax(1) if softmax else inputs.float()
        labels = F.one_hot(target.long(), self.n_classes).movedim(-1, 1).float()
        if labels.shape != probabilities.shape:
            raise ValueError(f"Logits/target shape mismatch: {inputs.shape}, {target.shape}")
        dims = tuple(range(2, inputs.ndim))
        scores = (2 * (probabilities * labels).sum(dims) + self.smooth) / (
            probabilities.square().sum(dims) + labels.sum(dims) + self.smooth
        )
        losses = 1 - scores
        if weight is not None:
            weights = torch.as_tensor(weight, device=inputs.device, dtype=torch.float32)
            if (weights.shape != (self.n_classes,) or not torch.isfinite(weights).all()
                    or (weights < 0).any() or weights.sum() <= 0):
                raise ValueError("Dice weights must be nonnegative with positive sum.")
            return (losses * weights).sum(1).mean() / weights.sum()
        return losses.mean()


class SegmentationLoss(nn.Module):
    """Combined Cross-Entropy, Soft Dice, and Pathology-Conditional AAR (Edema-Inclusive) Loss."""

    def __init__(self, n_classes=4, ce_weight=0.5, dice_weight=0.5, aar_weight=0.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.aar_weight = aar_weight
        self.dice = DiceLoss(n_classes)

    def forward(self, logits, target):
        # 1. Standard Unbiased Cross-Entropy
        ce = F.cross_entropy(logits.float(), target.long())

        # 2. Standard Unbiased Multi-Class Soft Dice
        dice = self.dice(logits, target, softmax=True)

        # 3. Conditional Area-at-Risk (AAR / Edema-Inclusive) Auxiliary Dice Loss
        # Active only when aar_weight > 0 and on slices containing actual pathology (Scar or Edema)
        # Uses symmetric linear denominator to balance precision and recall on the edema margin
        if self.aar_weight > 0.0:
            has_patho = ((target == 2) | (target == 3)).flatten(1).any(dim=1)
            if has_patho.any():
                probs = logits.float().softmax(1)
                p_aar = probs[has_patho, 2] + probs[has_patho, 3]
                y_aar = ((target[has_patho] == 2) | (target[has_patho] == 3)).float()
                dims = tuple(range(1, p_aar.ndim))
                inter = 2.0 * (p_aar * y_aar).sum(dims) + 1e-5
                denom = p_aar.sum(dims) + y_aar.sum(dims) + 1e-5
                aar_loss = (1.0 - inter / denom).mean()
            else:
                aar_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
        else:
            aar_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        total_loss = (
            self.ce_weight * ce
            + self.dice_weight * dice
            + self.aar_weight * aar_loss
        )
        return {
            "loss": total_loss,
            "ce": ce,
            "dice_loss": dice,
            "aar_loss": aar_loss,
        }
