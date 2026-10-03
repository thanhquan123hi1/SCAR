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
    """Combined Cross-Entropy, Soft Dice, Dual Pathology-Targeted, and Myocardial Wall Auxiliary Loss."""

    def __init__(
        self,
        n_classes=4,
        ce_weight=0.5,
        dice_weight=0.5,
        aar_weight=0.0,
        scar_weight=0.0,
        wall_weight=0.0,
        inclusion_weight=0.0,
        dice_class_weights=None,
        ce_class_weights=None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.aar_weight = aar_weight
        self.scar_weight = scar_weight
        self.wall_weight = wall_weight
        self.inclusion_weight = inclusion_weight
        self.dice_class_weights = dice_class_weights
        self.ce_class_weights = ce_class_weights
        self.dice = DiceLoss(n_classes)

    def forward(self, logits, target):
        if isinstance(logits, dict):
            wall_logits = logits.get("wall_logits")
            aar_stream_logits = logits.get("aar_logits")
            logits = logits["logits"]
        else:
            wall_logits = None
            aar_stream_logits = None

        # 1. Cross-Entropy (with optional class weights)
        if self.ce_class_weights is not None:
            ce_w = torch.as_tensor(self.ce_class_weights, device=logits.device, dtype=torch.float32)
            ce = F.cross_entropy(logits.float(), target.long(), weight=ce_w)
        else:
            ce = F.cross_entropy(logits.float(), target.long())

        # 2. Standard Unbiased Multi-Class Soft Dice (with optional class weighting)
        dice = self.dice(logits, target, weight=self.dice_class_weights, softmax=True)

        # 3. Conditional Area-at-Risk (AAR / Edema-Inclusive) Auxiliary Dice Loss
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

            # Extra supervision if model has explicit aar_stream_logits
            if aar_stream_logits is not None:
                y_aar_full = ((target == 2) | (target == 3)).float().unsqueeze(1)
                aar_bce = F.binary_cross_entropy_with_logits(aar_stream_logits, y_aar_full)
                p_stream_aar = torch.sigmoid(aar_stream_logits)
                dims = (2, 3)
                inter_stream = 2.0 * (p_stream_aar * y_aar_full).sum(dims) + 1e-5
                denom_stream = p_stream_aar.sum(dims) + y_aar_full.sum(dims) + 1e-5
                aar_stream_dice = (1.0 - inter_stream / denom_stream).mean()
                aar_loss = 0.5 * aar_loss + 0.5 * (0.5 * (aar_bce + aar_stream_dice))
        else:
            aar_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        # 4. Conditional Scar Auxiliary Dice Loss
        if self.scar_weight > 0.0:
            has_scar = (target == 3).flatten(1).any(dim=1)
            if has_scar.any():
                probs = logits.float().softmax(1)
                p_scar = probs[has_scar, 3]
                y_scar = (target[has_scar] == 3).float()
                dims = tuple(range(1, p_scar.ndim))
                inter_s = 2.0 * (p_scar * y_scar).sum(dims) + 1e-5
                denom_s = p_scar.sum(dims) + y_scar.sum(dims) + 1e-5
                scar_loss = (1.0 - inter_s / denom_s).mean()
            else:
                scar_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
        else:
            scar_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        # 5. Conditional Myocardial Wall Auxiliary Loss
        if self.wall_weight > 0.0 and wall_logits is not None:
            y_wall = (target > 0).float().unsqueeze(1)  # (B, 1, H, W)
            wall_bce = F.binary_cross_entropy_with_logits(wall_logits, y_wall)
            p_wall = torch.sigmoid(wall_logits)
            dims = (2, 3)
            inter_w = 2.0 * (p_wall * y_wall).sum(dims) + 1e-5
            denom_w = p_wall.sum(dims) + y_wall.sum(dims) + 1e-5
            wall_dice = (1.0 - inter_w / denom_w).mean()
            wall_loss = 0.5 * (wall_bce + wall_dice)
        else:
            wall_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        # 6. Hierarchical Pathology Topological Inclusion Loss (Scar in AAR in Myo)
        if self.inclusion_weight > 0.0:
            probs = logits.float().softmax(1)
            p_scar = probs[:, 3]
            p_aar = probs[:, 2] + probs[:, 3]
            p_myo = probs[:, 1] + p_aar
            # Scar cannot exceed AAR, AAR cannot exceed Myocardium
            violation = F.relu(p_scar - p_aar) + 0.5 * F.relu(p_aar - p_myo)
            inclusion_loss = violation.mean()
        else:
            inclusion_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        total_loss = (
            self.ce_weight * ce
            + self.dice_weight * dice
            + self.aar_weight * aar_loss
            + self.scar_weight * scar_loss
            + self.wall_weight * wall_loss
            + self.inclusion_weight * inclusion_loss
        )
        return {
            "loss": total_loss,
            "ce": ce,
            "dice_loss": dice,
            "aar_loss": aar_loss,
            "scar_loss": scar_loss,
            "wall_loss": wall_loss,
            "inclusion_loss": inclusion_loss,
        }
