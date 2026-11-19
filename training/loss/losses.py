"""Segmentation loss functions."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

# Version 4 adds I-MMSeg's fixed-class-count normalization.
LOSS_PROTOCOL_VERSION = 4


def _reduce_dice(losses, weight, n_classes, include_background, mask=None, normalization="weights"):
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
    denominator = weights.sum(1) if normalization == "weights" else weights.new_full((len(losses),), n_classes)
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
        normalization: str = "weights",
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
        if normalization not in {"weights", "classes"}:
            raise ValueError("Dice normalization must be weights or classes")
        self.normalization = normalization
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
            if self.normalization == "classes":
                # Match I-MMSeg's per-class operation order and divisor exactly.
                weights = [1.0] * self.n_classes if weight is None else weight
                if (len(weights) != self.n_classes or
                        any(not math.isfinite(float(v)) or v < 0 for v in weights) or sum(weights) <= 0):
                    raise ValueError("Dice weights must be finite, nonnegative with positive sum")
                loss = 0.0
                for index in range(0 if self.include_background else 1, self.n_classes):
                    score, label = probabilities[:, index], labels[:, index]
                    intersect = torch.sum(score * label)
                    y_sum = torch.sum(label * label)
                    z_sum = torch.sum(score * score)
                    dice = (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)
                    loss = loss + (1 - dice) * weights[index]
                return loss / self.n_classes
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
            return _reduce_dice(slice_losses, weight, self.n_classes, self.include_background, has_fg,
                                normalization=self.normalization)
        return _reduce_dice(slice_losses, weight, self.n_classes, self.include_background,
                            normalization=self.normalization)


BOUNDARY_DT_MODES = ("sym_bce", "karimi_mse", "fp_bce", "hybrid_karimi_farfp")


def _octagonal_outside_dist(
    mask: torch.Tensor,
    max_steps: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch octagonal approximation to 2D outside Euclidean distance (zero CPU sync).

    Returns ``(dist, has_fg)`` where ``dist`` is in ``[0, max_steps]`` (0 inside ``mask``)
    and ``has_fg`` has shape ``(B, C, 1, 1)`` indicating non-empty masks.
    """
    with torch.no_grad():
        m = mask.float()
        has_fg = (m.flatten(2).sum(dim=2) > 0).unsqueeze(-1).unsqueeze(-1)
        cur = m
        dist = torch.zeros_like(m)
        for step in range(max_steps):
            dist = dist + (1.0 - cur)
            if step % 2 == 0:
                cur = F.max_pool2d(cur, kernel_size=3, stride=1, padding=1)
            else:
                padded = F.pad(cur, (1, 1, 1, 1), mode="constant", value=0.0)
                cur = torch.maximum(
                    cur,
                    torch.maximum(
                        torch.maximum(padded[:, :, :-2, 1:-1], padded[:, :, 2:, 1:-1]),
                        torch.maximum(padded[:, :, 1:-1, :-2], padded[:, :, 1:-1, 2:]),
                    ),
                )
        return dist, has_fg


def _octagonal_outside_dt(
    mask: torch.Tensor,
    max_steps: int = 32,
    d_scale: float = 16.0,
    power: float = 2.0,
    empty_val: float = 1.0,
    margin: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Weighted outside distance map with optional dead-zone margin."""
    dist, has_fg = _octagonal_outside_dist(mask, max_steps=max_steps)
    eff_dist = (dist - margin).clamp_min(0.0)
    weight_map = (eff_dist / float(d_scale)).pow(float(power))
    return torch.where(has_fg, weight_map, torch.full_like(weight_map, float(empty_val)))


class SegmentationLoss(nn.Module):
    """Cross-Entropy and Soft Dice with optional hierarchical and distance-aware HD supervision."""

    def __init__(
        self,
        n_classes=4,
        ce_weight=0.5,
        dice_weight=0.5,
        aar_weight=0.0,
        scar_weight=0.0,
        anat_contrast_weight=0.0,
        hier_dice_weight=0.0,
        ring_tversky_beta=0.5,
        aar_tversky_beta=0.5,
        boundary_dt_weight=0.0,
        boundary_dt_mode="sym_bce",
        boundary_dt_power=2.0,
        boundary_dt_scale=16.0,
        boundary_dt_max_steps=32,
        boundary_dt_empty_val=1.0,
        boundary_dt_margin=0.0,
        dice_class_weights=None,
        ce_class_weights=None,
        pathology_reduction="positive",
        masked_negative_slices=False,
        negative_slice_handling=None,
        batch_dice=False,
        include_background=None,
        dice_normalization="weights",
    ):
        super().__init__()
        if pathology_reduction not in ("positive", "sample"):
            raise ValueError("pathology_reduction must be positive or sample")
        if boundary_dt_mode not in BOUNDARY_DT_MODES:
            raise ValueError(f"boundary_dt_mode must be one of {BOUNDARY_DT_MODES}")
        if not isinstance(boundary_dt_max_steps, int) or boundary_dt_max_steps < 1:
            raise ValueError("boundary_dt_max_steps must be a positive integer")
        if not math.isfinite(ring_tversky_beta) or not 0.0 < ring_tversky_beta < 1.0:
            raise ValueError("ring_tversky_beta must be strictly between 0 and 1")
        if not math.isfinite(aar_tversky_beta) or not 0.0 < aar_tversky_beta < 1.0:
            raise ValueError("aar_tversky_beta must be strictly between 0 and 1")
        self.pathology_reduction = pathology_reduction
        for name, value in (("ce_weight", ce_weight), ("dice_weight", dice_weight),
                            ("aar_weight", aar_weight), ("scar_weight", scar_weight),
                            ("anat_contrast_weight", anat_contrast_weight),
                            ("hier_dice_weight", hier_dice_weight),
                            ("boundary_dt_weight", boundary_dt_weight),
                            ("boundary_dt_empty_val", boundary_dt_empty_val),
                            ("boundary_dt_margin", boundary_dt_margin)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (("boundary_dt_power", boundary_dt_power),
                            ("boundary_dt_scale", boundary_dt_scale)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
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
        self.anat_contrast_weight = anat_contrast_weight
        self.hier_dice_weight = hier_dice_weight
        self.ring_tversky_beta = float(ring_tversky_beta)
        self.aar_tversky_beta = float(aar_tversky_beta)
        self.boundary_dt_weight = boundary_dt_weight
        self.boundary_dt_mode = boundary_dt_mode
        self.boundary_dt_power = float(boundary_dt_power)
        self.boundary_dt_scale = float(boundary_dt_scale)
        self.boundary_dt_max_steps = int(boundary_dt_max_steps)
        self.boundary_dt_empty_val = float(boundary_dt_empty_val)
        self.boundary_dt_margin = float(boundary_dt_margin)
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
            normalization=dice_normalization,
        )

    def validate_accumulation(self, accum_steps):
        if accum_steps > 1 and self.dice.batch_dice and self.dice_weight > 0:
            raise ValueError("Batch Dice requires accum_steps=1; use per-image Dice for gradient accumulation")
        if accum_steps > 1 and (self.hier_dice_weight > 0 or self.boundary_dt_weight > 0):
            raise ValueError("Batch-level hierarchical/distance losses require accum_steps=1")
        if (accum_steps > 1 and self.pathology_reduction == "positive"
                and (self.aar_weight > 0 or self.scar_weight > 0 or self.anat_contrast_weight > 0)):
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

        # 5. Bounded Batch-Level Composite Hierarchy & Edema-Scar Margin Loss
        if self.anat_contrast_weight > 0.0:
            probs = logits.float().softmax(1)
            p_myo = probs[:, 1]
            p_edema = probs[:, 2]
            p_scar = probs[:, 3]
            p_aar = p_edema + p_scar
            p_ring = p_myo + p_aar

            y_edema = (target == 2).float()
            y_scar = (target == 3).float()
            y_aar = y_edema + y_scar
            y_ring = (target > 0).float()

            # All-slice batch-level composite Dice terms (strictly bounded in [0, 1];
            # negative slices remain in the batch denominator to penalize false positives)
            batch_scar_dice = 1.0 - (2.0 * (p_scar * y_scar).sum() + 1e-5) / (p_scar.sum() + y_scar.sum() + 1e-5)
            batch_aar_dice = 1.0 - (2.0 * (p_aar * y_aar).sum() + 1e-5) / (p_aar.sum() + y_aar.sum() + 1e-5)
            batch_ring_dice = 1.0 - (2.0 * (p_ring * y_ring).sum() + 1e-5) / (p_ring.sum() + y_ring.sum() + 1e-5)

            # Strictly [0, 1)-bounded penalty for Scar false-positives inside GT Edema
            scar_in_edema_leak = (p_scar * y_edema).sum() / (y_edema.sum() + 1.0)

            anat_contrast_loss = (
                0.35 * batch_scar_dice
                + 0.35 * batch_aar_dice
                + 0.15 * batch_ring_dice
                + 0.15 * scar_in_edema_leak
            )
        else:
            anat_contrast_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        # 6. Clean Hierarchical Composite Batch-Dice (MyoRing, AAR/Edema-Incl, Scar)
        #    and Distance-Transform Hausdorff Surrogate Loss
        if self.hier_dice_weight > 0.0 or self.boundary_dt_weight > 0.0:
            probs = logits.float().softmax(1)
            # Compute positive and complementary negative region probabilities directly
            # from class sums to avoid 1 - (1 - eps) precision loss in float32/bf16.
            p_pos = torch.stack(
                (
                    probs[:, 1] + probs[:, 2] + probs[:, 3],  # Myocardial Ring {1,2,3}
                    probs[:, 2] + probs[:, 3],                 # Edema Inclusive / AAR {2,3}
                    probs[:, 3],                               # Scar {3}
                ),
                dim=1,
            )
            p_neg = torch.stack(
                (
                    probs[:, 0],                               # Outside Ring {0}
                    probs[:, 0] + probs[:, 1],                 # Outside AAR {0,1}
                    probs[:, 0] + probs[:, 1] + probs[:, 2],   # Outside Scar {0,1,2}
                ),
                dim=1,
            )
            y_pos = torch.stack(
                (
                    (target > 0),
                    (target == 2) | (target == 3),
                    (target == 3),
                ),
                dim=1,
            ).float()
            region_weights = torch.tensor((0.30, 0.35, 0.35), device=logits.device, dtype=torch.float32)

        if self.hier_dice_weight > 0.0:
            if abs(self.ring_tversky_beta - 0.5) < 1e-6 and abs(self.aar_tversky_beta - 0.5) < 1e-6:
                inter_h = 2.0 * (p_pos * y_pos).sum(dim=(0, 2, 3)) + 1e-5
                denom_h = p_pos.square().sum(dim=(0, 2, 3)) + y_pos.sum(dim=(0, 2, 3)) + 1e-5
                per_reg_h = 1.0 - inter_h / denom_h
            else:
                def _region_tversky_or_dice(idx: int, beta: float) -> torch.Tensor:
                    if abs(beta - 0.5) < 1e-6:
                        inter = 2.0 * (p_pos[:, idx] * y_pos[:, idx]).sum() + 1e-5
                        denom = p_pos[:, idx].square().sum() + y_pos[:, idx].sum() + 1e-5
                        return 1.0 - inter / denom
                    alpha = 1.0 - beta
                    tp = (p_pos[:, idx] * y_pos[:, idx]).sum()
                    fp = (p_pos[:, idx] * (1.0 - y_pos[:, idx])).sum()
                    fn = (p_neg[:, idx] * y_pos[:, idx]).sum()
                    return 1.0 - (tp + 1e-5) / (tp + alpha * fp + beta * fn + 1e-5)

                ring_loss = _region_tversky_or_dice(0, self.ring_tversky_beta)
                aar_h_loss = _region_tversky_or_dice(1, self.aar_tversky_beta)
                scar_h_loss = _region_tversky_or_dice(2, 0.5)
                per_reg_h = torch.stack((ring_loss, aar_h_loss, scar_h_loss), dim=0)
            hier_dice_loss = (per_reg_h * region_weights).sum()
        else:
            hier_dice_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        if self.boundary_dt_weight > 0.0:
            y_neg = 1.0 - y_pos
            fg_denom = y_pos.sum(dim=(0, 2, 3)) + p_pos.detach().sum(dim=(0, 2, 3)) + 1.0
            if self.boundary_dt_mode == "fp_bce":
                w_gt = _octagonal_outside_dt(
                    y_pos,
                    max_steps=self.boundary_dt_max_steps,
                    d_scale=self.boundary_dt_scale,
                    power=self.boundary_dt_power,
                    empty_val=self.boundary_dt_empty_val,
                    margin=self.boundary_dt_margin,
                )
                fp_err = (-torch.log(p_neg.clamp(min=1e-6))).clamp(max=10.0)
                per_reg = (w_gt * y_neg * fp_err).sum(dim=(0, 2, 3)) / fg_denom
            elif self.boundary_dt_mode == "hybrid_karimi_farfp":
                pred_cls = logits.detach().argmax(1)
                pred_pos = torch.stack(
                    (
                        (pred_cls > 0),
                        (pred_cls == 2) | (pred_cls == 3),
                        (pred_cls == 3),
                    ),
                    dim=1,
                ).float()
                dist_both, has_fg_both = _octagonal_outside_dist(
                    torch.cat((y_pos, pred_pos), dim=1),
                    max_steps=self.boundary_dt_max_steps,
                )
                d_gt, h_gt = dist_both[:, :3], has_fg_both[:, :3]
                d_pr, h_pr = dist_both[:, 3:], has_fg_both[:, 3:]

                w_gt = torch.where(
                    h_gt,
                    (d_gt / self.boundary_dt_scale).pow(self.boundary_dt_power),
                    torch.full_like(d_gt, 1.0),
                )
                w_pr = torch.where(
                    h_pr,
                    (d_pr / self.boundary_dt_scale).pow(self.boundary_dt_power),
                    torch.full_like(d_pr, 1.0),
                )
                # Dead-zone margin: half-margin on thin MyoRing, full margin on AAR/Scar
                # so near-boundary Dice calibration is untouched while distant FPs get crushed.
                margins = torch.tensor(
                    (0.5 * self.boundary_dt_margin, self.boundary_dt_margin, self.boundary_dt_margin),
                    device=logits.device,
                    dtype=torch.float32,
                ).view(1, 3, 1, 1)
                w_far = torch.where(
                    h_gt,
                    ((d_gt - margins).clamp_min(0.0) / self.boundary_dt_scale).pow(self.boundary_dt_power),
                    torch.full_like(d_gt, self.boundary_dt_empty_val),
                )
                karimi_reg = (y_neg * p_pos.square() * w_gt + y_pos * p_neg.square() * w_pr).sum(dim=(0, 2, 3)) / fg_denom
                fp_err = (-torch.log(p_neg.clamp(min=1e-6))).clamp(max=10.0)
                far_fp_reg = (w_far * y_neg * fp_err).sum(dim=(0, 2, 3)) / fg_denom
                per_reg = karimi_reg + 0.30 * far_fp_reg
                region_weights = torch.tensor((0.30, 0.40, 0.30), device=logits.device, dtype=torch.float32)
            else:
                pred_cls = logits.detach().argmax(1)
                pred_pos = torch.stack(
                    (
                        (pred_cls > 0),
                        (pred_cls == 2) | (pred_cls == 3),
                        (pred_cls == 3),
                    ),
                    dim=1,
                ).float()
                both_dt = _octagonal_outside_dt(
                    torch.cat((y_pos, pred_pos), dim=1),
                    max_steps=self.boundary_dt_max_steps,
                    d_scale=self.boundary_dt_scale,
                    power=self.boundary_dt_power,
                    empty_val=self.boundary_dt_empty_val,
                    margin=self.boundary_dt_margin,
                )
                w_gt, w_pr = both_dt[:, :3], both_dt[:, 3:]
                if self.boundary_dt_mode == "sym_bce":
                    fp_err = (-torch.log(p_neg.clamp(min=1e-6))).clamp(max=10.0)
                    fn_err = (-torch.log(p_pos.clamp(min=1e-6))).clamp(max=10.0)
                    per_reg = (w_gt * y_neg * fp_err + w_pr * y_pos * fn_err).sum(dim=(0, 2, 3)) / fg_denom
                else:  # karimi_mse
                    err_map = y_neg * p_pos.square() * w_gt + y_pos * p_neg.square() * w_pr
                    per_reg = err_map.sum(dim=(0, 2, 3)) / fg_denom
            boundary_dt_loss = (per_reg * region_weights).sum()
        else:
            boundary_dt_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        total_loss = (
            self.ce_weight * ce
            + self.dice_weight * dice
            + self.aar_weight * aar_loss
            + self.scar_weight * scar_loss
            + self.anat_contrast_weight * anat_contrast_loss
            + self.hier_dice_weight * hier_dice_loss
            + self.boundary_dt_weight * boundary_dt_loss
        )
        result = {
            "loss": total_loss,
            "ce": ce,
            "dice_loss": dice,
            "aar_loss": aar_loss,
            "scar_loss": scar_loss,
            "anat_contrast_loss": anat_contrast_loss,
        }
        if self.hier_dice_weight > 0.0 or self.boundary_dt_weight > 0.0:
            result["hier_dice_loss"] = hier_dice_loss
            result["boundary_dt_loss"] = boundary_dt_loss
        return result


__all__ = [
    "DiceLoss",
    "SegmentationLoss",
    "MaskedSliceDiceLoss",
    "BatchSoftDiceLoss",
    "BOUNDARY_DT_MODES",
    "validate_microlesion_resolution",
    "LOSS_PROTOCOL_VERSION",
]
