"""Loss functions for LGE Cardiac MRI Segmentation.

Provides:
- OneVsRestCompoundLoss: BCE (with positive weighting & focal modulation) + Binary SoftDice
- FocalTverskyLoss: Asymmetric penalty for false negatives (recall boost for rare scar)
- SoftDiceLoss / Multi-class Dice Loss
- Weighted Cross-Entropy + Dice Loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masks_to_one_vs_rest(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer mask (B, H, W) or (B, D, H, W) to foreground one-vs-rest binary targets.
    
    Output shape: (B, num_classes - 1, ...) where channel k corresponds to class_id = k + 1.
    """
    targets = []
    for class_id in range(1, num_classes):
        targets.append((mask == class_id).float())
    if not targets:
        shape = (mask.shape[0], 0) + tuple(mask.shape[1:])
        return torch.zeros(shape, device=mask.device, dtype=torch.float32)
    return torch.stack(targets, dim=1)


class SoftDiceLoss(nn.Module):
    """Multi-class Soft Dice Loss with Laplace smoothing (smooth=1.0)."""

    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_background: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(labels.long(), num_classes=self.num_classes).movedim(-1, 1).float()
        
        start_cls = 1 if self.ignore_background else 0
        dims = tuple(range(2, probs.ndim))
        
        p = probs[:, start_cls:]
        t = one_hot[:, start_cls:]
        intersection = torch.sum(p * t, dim=dims)
        denominator = torch.sum(p + t, dim=dims).clamp_min(1e-6)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return (1.0 - dice).mean()


class OneVsRestCompoundLoss(nn.Module):
    """Compound loss for independent foreground binary heads (One-vs-Rest).
    
    Total = bce_weight * (BCE with pos_weight + Focal modulation) + dice_weight * BinarySoftDice
    """

    def __init__(
        self,
        num_classes: int,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_weight: float = 0.4,
        focal_gamma: float = 2.0,
        pos_weight: list[float] | torch.Tensor | None = None,
        class_weights: list[float] | torch.Tensor | None = None,
        smooth: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.smooth = smooth

        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def _binary_dice_loss(self, fg_logits: torch.Tensor, fg_target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(fg_logits)
        K = probs.shape[1]
        if K == 0:
            return torch.tensor(0.0, device=probs.device)

        dims = tuple(range(2, probs.ndim))
        intersection = (probs * fg_target).sum(dim=dims)
        union = (probs + fg_target).sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, K)

        if self.class_weights is not None:
            cw = self.class_weights.to(dice.device)
            if cw.numel() >= K:
                cw = cw[:K]
            elif cw.numel() == 1 and K > 1:
                cw = cw.repeat(K)
            if cw.numel() == K:
                w = cw.view(1, -1)
                return 1.0 - (dice * w).sum() / (w.sum().clamp_min(1e-8) * dice.shape[0])

        return 1.0 - dice.mean()

    def _bce_or_focal(self, fg_logits: torch.Tensor, fg_target: torch.Tensor) -> torch.Tensor:
        loss_map = F.binary_cross_entropy_with_logits(fg_logits, fg_target, reduction="none")
        K = fg_logits.shape[1]

        if self.pos_weight is not None:
            pw = self.pos_weight.to(fg_logits.device)
            if pw.numel() >= K:
                pw = pw[:K]
            elif pw.numel() == 1 and K > 1:
                pw = pw.repeat(K)
            if pw.numel() == K:
                pw_view = pw.view(1, -1, *([1] * (fg_logits.ndim - 2)))
                pos_mask = (fg_target > 0.5).float()
                weight_map = 1.0 + pos_mask * (pw_view - 1.0)
                loss_map = loss_map * weight_map

        if self.focal_weight > 0.0:
            probs = torch.sigmoid(fg_logits)
            pt = torch.where(fg_target > 0.5, probs, 1.0 - probs)
            focal = (1.0 - pt).pow(self.focal_gamma)
            loss_map = (1.0 - self.focal_weight) * loss_map + self.focal_weight * (focal * loss_map)

        if self.class_weights is not None:
            cw = self.class_weights.to(fg_logits.device)
            if cw.numel() >= K:
                cw = cw[:K]
            elif cw.numel() == 1 and K > 1:
                cw = cw.repeat(K)
            if cw.numel() == K:
                cw_view = cw.view(1, -1, *([1] * (fg_logits.ndim - 2)))
                loss_map = loss_map * cw_view
                spatial_voxels = fg_logits.shape[0] * torch.prod(torch.tensor(fg_logits.shape[2:], device=fg_logits.device))
                return loss_map.sum() / (cw.sum().clamp_min(1e-8) * spatial_voxels)

        return loss_map.mean()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == self.num_classes - 1:
            fg_logits = logits
        elif logits.shape[1] == self.num_classes:
            fg_logits = logits[:, 1:]
        else:
            fg_logits = logits

        if labels.ndim == logits.ndim - 1:
            fg_target = masks_to_one_vs_rest(labels, self.num_classes)
        else:
            fg_target = labels.float()

        bce = self._bce_or_focal(fg_logits, fg_target)
        dice = self._binary_dice_loss(fg_logits, fg_target)
        return self.bce_weight * bce + self.dice_weight * dice


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for imbalanced medical image segmentation."""

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == self.num_classes:
            probs = torch.softmax(logits, dim=1)[:, 1:]
        else:
            probs = torch.sigmoid(logits)

        if labels.ndim == logits.ndim - 1:
            targets = masks_to_one_vs_rest(labels, self.num_classes)
        else:
            targets = labels.float()

        dims = tuple(range(2, probs.ndim))
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1.0 - tversky).pow(self.gamma)
        return focal_tversky.mean()


def ce_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    class_weights: list[float] | torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Weighted Cross-Entropy + Soft Dice Loss."""
    weight = None
    if class_weights is not None:
        weight = torch.as_tensor(class_weights, dtype=torch.float32, device=logits.device)
    ce = F.cross_entropy(logits, labels.long(), weight=weight)
    dice_fn = SoftDiceLoss(num_classes=num_classes).to(logits.device)
    dice = dice_fn(logits, labels)
    return ce_weight * ce + dice_weight * dice


LOSS_REGISTRY = {
    "ce": lambda logits, labels, **kw: F.cross_entropy(logits, labels.long()),
    "dice": lambda logits, labels, num_classes=5, **kw: SoftDiceLoss(num_classes).to(logits.device)(logits, labels),
    "ce_dice": ce_dice_loss,
    "one_vs_rest_compound": OneVsRestCompoundLoss,
    "focal_tversky": FocalTverskyLoss,
}


def build_loss(name: str, **kwargs):
    """Factory function for loss modules."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}")
    
    cls_or_fn = LOSS_REGISTRY[name]
    if isinstance(cls_or_fn, type):
        return cls_or_fn(**kwargs)
    return lambda logits, labels: cls_or_fn(logits, labels, **kwargs)
