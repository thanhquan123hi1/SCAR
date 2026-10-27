"""Segmentation loss functions."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

# Version 3 fixes foreground selection and uses per-image masked Dice reduction.
LOSS_PROTOCOL_VERSION = 3


def _reduce_dice(losses, weight, n_classes, include_background, mask=None):
    """Mean per-image class losses, with differentiable zero for empty images."""
    weights = torch.ones(n_classes, device=losses.device, dtype=torch.float32)
    if weight is not None:
        weights = torch.as_tensor(weight, device=losses.device, dtype=torch.float32)
        if (weights.shape != (n_classes,) or not torch.isfinite(weights).all()
                or (weights < 0).any() or not torch.isfinite(weights.sum()) or weights.sum() <= 0):
            raise ValueError("Dice weights must be finite, nonnegative with positive sum.")
    start = 0 if include_background else 1
    selected = losses[:, start:]
    weights = weights[start:].unsqueeze(0).expand_as(selected)
    if mask is not None:
        weights = weights * mask[:, start:]
    denominator = weights.sum(1)
    return ((selected * weights).sum(1) / denominator.masked_fill(denominator == 0, 1)).mean()


def validate_microlesion_resolution(
    img_size: int | tuple[int, int],
    min_resolution: int = 192,
    strict: bool = True,
) -> bool:
    """Check an optional configured size limit, not anatomical preservation.

    Pixel dimensions alone cannot establish preservation: spacing, resampling
    scale and crop coverage must be evaluated separately.
    """
    h = img_size if isinstance(img_size, int) else img_size[0]
    w = img_size if isinstance(img_size, int) else img_size[1]
    if min(h, w) < min_resolution:
        if strict:
            raise ValueError(
                f"Resolution ({h}x{w}) is below the configured minimum "
                f"{min_resolution}x{min_resolution}."
            )
        return False
    return True


class MaskedSliceDiceLoss(nn.Module):
    """Slice-wise Dice loss with foreground-presence masking (Finding G-01).

    For each class c, loss on slice i is only accumulated if class c is actually
    present in the ground truth for that slice (has_fg > 0).
    Reduces over classes within each image, then images. Empty images contribute
    differentiable zero; CE is still needed to penalize their false positives.
    """

    def __init__(
        self,
        n_classes: int = 4,
        smooth: float = 1e-5,
        include_background: bool = False,
    ):
        super().__init__()
        if not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError("n_classes must be an integer >= 2")
        if not math.isfinite(smooth) or smooth <= 0:
            raise ValueError("Dice smooth must be finite and positive")
        self.n_classes = n_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        weight: Any = None,
        softmax: bool = True,
    ) -> torch.Tensor:
        probabilities = inputs.float().softmax(1) if softmax else inputs.float()
        labels = F.one_hot(target.long(), self.n_classes).movedim(-1, 1).float()
        if labels.shape != probabilities.shape:
            raise ValueError(f"Logits/target shape mismatch: {inputs.shape}, {target.shape}")
        dims = tuple(range(2, inputs.ndim))
        scores = (2 * (probabilities * labels).sum(dims) + self.smooth) / (
            probabilities.square().sum(dims) + labels.sum(dims) + self.smooth
        )
        slice_losses = 1 - scores  # shape (B, C)

        has_fg = (labels.sum(dims) > 0).float()  # shape (B, C)
        return _reduce_dice(slice_losses, weight, self.n_classes, self.include_background, has_fg)


class BatchSoftDiceLoss(nn.Module):
    """Batch-level Soft Dice loss (Finding G-01 alternative).

    Instead of reducing over 2D slice spatial dimensions independently,
    reductions are computed across the entire batch (B, H, W).
    This aggregates gradient evidence across slices, naturally suppressing
    spikes from isolated empty slices in 3D volumes.
    """

    def __init__(
        self,
        n_classes: int = 4,
        smooth: float = 1e-5,
        include_background: bool = False,
    ):
        super().__init__()
        if not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError("n_classes must be an integer >= 2")
        if not math.isfinite(smooth) or smooth <= 0:
            raise ValueError("Dice smooth must be finite and positive")
        self.n_classes = n_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        weight: Any = None,
        softmax: bool = True,
    ) -> torch.Tensor:
        probabilities = inputs.float().softmax(1) if softmax else inputs.float()
        labels = F.one_hot(target.long(), self.n_classes).movedim(-1, 1).float()
        if labels.shape != probabilities.shape:
            raise ValueError(f"Logits/target shape mismatch: {inputs.shape}, {target.shape}")

        dims = (0,) + tuple(range(2, inputs.ndim))
        inter = 2 * (probabilities * labels).sum(dims) + self.smooth
        denom = probabilities.square().sum(dims) + labels.sum(dims) + self.smooth
        class_losses = 1 - (inter / denom)  # shape (C,)

        return _reduce_dice(class_losses.unsqueeze(0), weight, self.n_classes, self.include_background)


class DiceLoss(nn.Module):
    """Per-image equally weighted class Dice with squared denominator.

    Includes background by default; reductions are float32 even under autocast. Per-image
    reduction makes sample-weighted gradient accumulation well-defined.

    Supports negative slice masking (masked_negative_slices=True or negative_slice_handling='mask')
    and batch-level aggregation (batch_dice=True or negative_slice_handling='batch')
    with explicit, checkpointed reduction semantics.
    """

    def __init__(
        self,
        n_classes: int = 4,
        smooth: float = 1e-5,
        masked_negative_slices: bool = False,
        negative_slice_handling: str | None = None,
        batch_dice: bool = False,
        include_background: bool | None = None,
    ):
        super().__init__()
        if not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError("n_classes must be an integer >= 2")
        if not math.isfinite(smooth) or smooth <= 0:
            raise ValueError("Dice smooth must be finite and positive")
        if negative_slice_handling not in (None, "mask", "batch", "none"):
            raise ValueError("negative_slice_handling must be None, 'mask', 'batch', or 'none'")
        self.n_classes = n_classes
        self.smooth = smooth
        self.masked_negative_slices = masked_negative_slices or (negative_slice_handling == "mask")
        self.batch_dice = batch_dice or (negative_slice_handling == "batch")
        if self.batch_dice and self.masked_negative_slices:
            raise ValueError("Choose either batch Dice or masked per-image Dice")
        self.negative_slice_handling = negative_slice_handling
        if include_background is None:
            self.include_background = False if (self.masked_negative_slices or self.batch_dice) else True
        else:
            self.include_background = include_background

    def forward(self, inputs, target, weight=None, softmax=False):
        probabilities = inputs.float().softmax(1) if softmax else inputs.float()
        labels = F.one_hot(target.long(), self.n_classes).movedim(-1, 1).float()
        if labels.shape != probabilities.shape:
            raise ValueError(f"Logits/target shape mismatch: {inputs.shape}, {target.shape}")

        if self.batch_dice:
            dims = (0,) + tuple(range(2, inputs.ndim))
            inter = 2 * (probabilities * labels).sum(dims) + self.smooth
            denom = probabilities.square().sum(dims) + labels.sum(dims) + self.smooth
            losses = 1 - (inter / denom)
            return _reduce_dice(losses.unsqueeze(0), weight, self.n_classes, self.include_background)

        dims = tuple(range(2, inputs.ndim))
        scores = (2 * (probabilities * labels).sum(dims) + self.smooth) / (
            probabilities.square().sum(dims) + labels.sum(dims) + self.smooth
        )
        slice_losses = 1 - scores

        if self.masked_negative_slices:
            has_fg = (labels.sum(dims) > 0).float()
            return _reduce_dice(slice_losses, weight, self.n_classes, self.include_background, has_fg)
        return _reduce_dice(slice_losses, weight, self.n_classes, self.include_background)


class SegmentationLoss(nn.Module):
    """Cross-Entropy and Soft Dice with optional AAR and scar supervision."""

    def __init__(
        self,
        n_classes=4,
        ce_weight=0.5,
        dice_weight=0.5,
        aar_weight=0.0,
        scar_weight=0.0,
        dice_class_weights=None,
        ce_class_weights=None,
        pathology_reduction="positive",
        masked_negative_slices=False,
        negative_slice_handling=None,
        batch_dice=False,
        include_background=None,
    ):
        super().__init__()
        if pathology_reduction not in ("positive", "sample"):
            raise ValueError("pathology_reduction must be positive or sample")
        self.pathology_reduction = pathology_reduction
        for name, value in (("ce_weight", ce_weight), ("dice_weight", dice_weight),
                            ("aar_weight", aar_weight), ("scar_weight", scar_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, values in (("dice_class_weights", dice_class_weights), ("ce_class_weights", ce_class_weights)):
            if values is not None:
                weights = torch.as_tensor(values, dtype=torch.float32)
                if (weights.shape != (n_classes,) or not torch.isfinite(weights).all()
                        or (weights < 0).any() or not torch.isfinite(weights.sum()) or weights.sum() <= 0):
                    raise ValueError(f"{name} must have {n_classes} finite nonnegative weights with positive sum")
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.aar_weight = aar_weight
        self.scar_weight = scar_weight
        self.dice_class_weights = dice_class_weights
        self.ce_class_weights = ce_class_weights
        self.masked_negative_slices = masked_negative_slices
        self.negative_slice_handling = negative_slice_handling
        self.batch_dice = batch_dice
        self.dice = DiceLoss(
            n_classes=n_classes,
            masked_negative_slices=masked_negative_slices,
            negative_slice_handling=negative_slice_handling,
            batch_dice=batch_dice,
            include_background=include_background,
        )

    def validate_accumulation(self, accum_steps):
        if accum_steps > 1 and self.dice.batch_dice and self.dice_weight > 0:
            raise ValueError("Batch Dice requires accum_steps=1; use per-image Dice for gradient accumulation")
        if (accum_steps > 1 and self.pathology_reduction == "positive"
                and (self.aar_weight > 0 or self.scar_weight > 0)):
            raise ValueError("Auxiliary positive reduction requires accum_steps=1; use pathology_reduction='sample'")

    def forward(self, logits, target):
        # 1. Cross-Entropy (with optional class weights)
        if self.ce_class_weights is not None:
            ce_w = torch.as_tensor(self.ce_class_weights, device=logits.device, dtype=torch.float32)
            pixel_loss = F.cross_entropy(logits.float(), target.long(), weight=ce_w, reduction="none")
            numerator = pixel_loss.flatten(1).sum(1)
            denominator = ce_w[target.long()].flatten(1).sum(1)
            # Per-image reduction preserves the trainer's sample-weighted accumulation.
            # A slice containing only ignored (zero-weight) classes contributes zero CE.
            ce = (numerator / denominator.masked_fill(denominator == 0, 1)).mean()
        else:
            ce = F.cross_entropy(logits.float(), target.long())

        # 2. Multi-Class Soft Dice (with optional class weighting and negative-slice handling)
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
                if self.pathology_reduction == "sample":
                    aar_loss = aar_loss * has_patho.float().mean()
            else:
                aar_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
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
                if self.pathology_reduction == "sample":
                    scar_loss = scar_loss * has_scar.float().mean()
            else:
                scar_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
        else:
            scar_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        total_loss = (
            self.ce_weight * ce
            + self.dice_weight * dice
            + self.aar_weight * aar_loss
            + self.scar_weight * scar_loss
        )
        return {
            "loss": total_loss,
            "ce": ce,
            "dice_loss": dice,
            "aar_loss": aar_loss,
            "scar_loss": scar_loss,
        }


__all__ = [
    "DiceLoss",
    "SegmentationLoss",
    "MaskedSliceDiceLoss",
    "BatchSoftDiceLoss",
    "validate_microlesion_resolution",
    "LOSS_PROTOCOL_VERSION",
]
