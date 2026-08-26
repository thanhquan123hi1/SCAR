"""Loss functions for segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss (mean over foreground classes)."""
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(labels.long(), num_classes=num_classes).movedim(-1, 1).float()
    dims = tuple(i for i in range(probs.ndim) if i != 1)
    intersection = torch.sum(probs[:, 1:] * one_hot[:, 1:], dim=dims)
    denominator = torch.sum(probs[:, 1:] + one_hot[:, 1:], dim=dims).clamp_min(smooth)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return (1.0 - dice).mean()


def ce_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy + Dice loss."""
    ce = F.cross_entropy(logits, labels.long())
    dice = dice_loss(logits, labels, num_classes=num_classes)
    return ce_weight * ce + dice_weight * dice


LOSS_REGISTRY = {
    "ce": lambda logits, labels, **kw: F.cross_entropy(logits, labels.long()),
    "dice": dice_loss,
    "ce_dice": ce_dice_loss,
}


def build_loss(name: str, **kwargs):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}")
    return lambda logits, labels: LOSS_REGISTRY[name](logits, labels, **kwargs)
