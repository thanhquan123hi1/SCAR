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


BOUNDARY_DT_MODES = ("sym_bce", "karimi_mse", "karimi_qlog", "karimi_fp_qlog", "fp_bce", "hybrid_karimi_farfp")


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
    d_scale: float | torch.Tensor = 16.0,
    power: float = 2.0,
    empty_val: float | torch.Tensor = 1.0,
    margin: float | torch.Tensor = 0.0,
    max_norm: float | None = None,
) -> torch.Tensor:
    """Weighted outside distance map with optional dead-zone margin and norm clamp."""
    dist, has_fg = _octagonal_outside_dist(mask, max_steps=max_steps)
    eff_dist = (dist - margin).clamp_min(0.0)
    scale = d_scale if isinstance(d_scale, torch.Tensor) else float(d_scale)
    norm_dist = eff_dist / scale
    if max_norm is not None:
        norm_dist = norm_dist.clamp(max=float(max_norm))
    weight_map = norm_dist.pow(float(power))
    if isinstance(empty_val, torch.Tensor):
        empty_map = empty_val.to(device=weight_map.device, dtype=weight_map.dtype).expand_as(weight_map)
    else:
        empty_map = torch.full_like(weight_map, float(empty_val))
    return torch.where(has_fg, weight_map, empty_map)


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
        slice_hier_weight=0.0,
        hier_region_weights=None,
        ring_tversky_beta=0.5,
        aar_tversky_beta=0.5,
        boundary_dt_weight=0.0,
        boundary_dt_slice_weight=0.0,
        boundary_dt_mode="sym_bce",
        boundary_dt_power=2.0,
        boundary_dt_scale=16.0,
        boundary_dt_scales=None,
        boundary_dt_region_weights=None,
        boundary_dt_max_steps=32,
        boundary_dt_empty_val=1.0,
        boundary_dt_empty_vals=None,
        boundary_dt_margin=0.0,
        boundary_dt_max_norm=None,
        boundary_dt_myo_fp_alpha=0.0,
        norm_myo_dt_weight=0.0,
        intra_myo_far_fp_weight=0.0,
        empty_slice_fp_weight=0.0,
        mod_aux_weight=0.0,
        mod_contrast_weight=0.0,
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
                            ("slice_hier_weight", slice_hier_weight),
                            ("boundary_dt_weight", boundary_dt_weight),
                            ("boundary_dt_slice_weight", boundary_dt_slice_weight),
                            ("boundary_dt_empty_val", boundary_dt_empty_val),
                            ("boundary_dt_margin", boundary_dt_margin),
                            ("boundary_dt_myo_fp_alpha", boundary_dt_myo_fp_alpha),
                            ("norm_myo_dt_weight", norm_myo_dt_weight),
                            ("intra_myo_far_fp_weight", intra_myo_far_fp_weight),
                            ("empty_slice_fp_weight", empty_slice_fp_weight),
                            ("mod_aux_weight", mod_aux_weight),
                            ("mod_contrast_weight", mod_contrast_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if boundary_dt_slice_weight > 1.0:
            raise ValueError("boundary_dt_slice_weight must be between 0.0 and 1.0")
        for name, value in (("boundary_dt_power", boundary_dt_power),
                            ("boundary_dt_scale", boundary_dt_scale)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if boundary_dt_max_norm is not None:
            if not math.isfinite(boundary_dt_max_norm) or boundary_dt_max_norm <= 0:
                raise ValueError("boundary_dt_max_norm must be finite and positive")
        if boundary_dt_scales is not None:
            scales_t = torch.as_tensor(boundary_dt_scales, dtype=torch.float32)
            if scales_t.shape != (3,) or not torch.isfinite(scales_t).all() or (scales_t <= 0).any():
                raise ValueError("boundary_dt_scales must have 3 finite positive values [ring, aar, scar]")
        if boundary_dt_empty_vals is not None:
            ev_t = torch.as_tensor(boundary_dt_empty_vals, dtype=torch.float32)
            if ev_t.shape != (3,) or not torch.isfinite(ev_t).all() or (ev_t < 0).any():
                raise ValueError("boundary_dt_empty_vals must have 3 finite nonnegative values [ring, aar, scar]")
        for name, values in (("hier_region_weights", hier_region_weights),
                             ("boundary_dt_region_weights", boundary_dt_region_weights)):
            if values is not None:
                rw = torch.as_tensor(values, dtype=torch.float32)
                if (rw.shape != (3,) or not torch.isfinite(rw).all()
                        or (rw < 0).any() or not torch.isfinite(rw.sum()) or rw.sum() <= 0):
                    raise ValueError(f"{name} must have 3 finite nonnegative weights with positive sum")
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
        self.slice_hier_weight = float(slice_hier_weight)
        self.hier_region_weights = [float(v) for v in hier_region_weights] if hier_region_weights is not None else None
        self.ring_tversky_beta = float(ring_tversky_beta)
        self.aar_tversky_beta = float(aar_tversky_beta)
        self.boundary_dt_weight = boundary_dt_weight
        self.boundary_dt_slice_weight = float(boundary_dt_slice_weight)
        self.boundary_dt_mode = boundary_dt_mode
        self.boundary_dt_power = float(boundary_dt_power)
        self.boundary_dt_scale = float(boundary_dt_scale)
        self.boundary_dt_scales = [float(v) for v in boundary_dt_scales] if boundary_dt_scales is not None else None
        self.boundary_dt_region_weights = (
            [float(v) for v in boundary_dt_region_weights] if boundary_dt_region_weights is not None else None
        )
        self.boundary_dt_max_steps = int(boundary_dt_max_steps)
        self.boundary_dt_empty_val = float(boundary_dt_empty_val)
        self.boundary_dt_empty_vals = (
            [float(v) for v in boundary_dt_empty_vals] if boundary_dt_empty_vals is not None else None
        )
        self.boundary_dt_margin = float(boundary_dt_margin)
        self.boundary_dt_max_norm = float(boundary_dt_max_norm) if boundary_dt_max_norm is not None else None
        self.boundary_dt_myo_fp_alpha = float(boundary_dt_myo_fp_alpha)
        self.norm_myo_dt_weight = float(norm_myo_dt_weight)
        self.intra_myo_far_fp_weight = float(intra_myo_far_fp_weight)
        self.empty_slice_fp_weight = float(empty_slice_fp_weight)
        self.mod_aux_weight = float(mod_aux_weight)
        self.mod_contrast_weight = float(mod_contrast_weight)
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
        if accum_steps > 1 and (
            self.hier_dice_weight > 0
            or self.boundary_dt_weight > 0
            or self.norm_myo_dt_weight > 0
            or self.intra_myo_far_fp_weight > 0
            or self.empty_slice_fp_weight > 0
            or self.mod_aux_weight > 0
            or self.mod_contrast_weight > 0
        ):
            raise ValueError("Batch-level hierarchical/distance/modality losses require accum_steps=1")
        if (accum_steps > 1 and self.pathology_reduction == "positive"
                and (self.aar_weight > 0 or self.scar_weight > 0 or self.anat_contrast_weight > 0)):
            raise ValueError("Auxiliary positive reduction requires accum_steps=1; use pathology_reduction='sample'")

    @staticmethod
    def _aux_ce_dice(logits: torch.Tensor, target: torch.Tensor, class_weights: tuple[float, ...]) -> torch.Tensor:
        logits_f = logits.float()
        target_l = target.long()
        ce = F.cross_entropy(logits_f, target_l)
        probs = logits_f.softmax(1)
        dice_sum = 0.0
        n_cls = len(class_weights)
        for c, w in enumerate(class_weights):
            p = probs[:, c]
            y = (target_l == c).float()
            inter = 2.0 * (p * y).sum() + 1e-5
            denom = p.square().sum() + y.sum() + 1e-5
            dice_sum = dice_sum + w * (1.0 - inter / denom)
        dice = dice_sum / n_cls
        return 0.4 * ce + 0.6 * dice

    @staticmethod
    def _supcon_loss(anchors: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        m = anchors.shape[0]
        if m < 2:
            return anchors.sum() * 0.0
        sim = torch.matmul(anchors, anchors.T) / temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        eye = torch.eye(m, device=anchors.device, dtype=torch.bool)
        pos_mask = labels.unsqueeze(1).eq(labels.unsqueeze(0)) & (~eye)
        valid = pos_mask.any(dim=1)
        if not valid.any():
            return anchors.sum() * 0.0
        log_denom = torch.logsumexp(sim.masked_fill(eye, -1e9), dim=1, keepdim=True)
        log_prob = sim - log_denom
        pos_cnt = pos_mask.float().sum(dim=1).clamp_min(1.0)
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_cnt
        return -mean_log_prob_pos[valid].mean()

    def _compute_modality_contrastive(
        self,
        mod_aux: dict,
        target: torch.Tensor,
        n_view_list: tuple[int, int, int] = (32, 16, 8),
    ) -> torch.Tensor:
        """I-MMSeg-inspired biology-aware multi-scale supervised contrastive alignment.

        Canonical target labels: 0=bg, 1=normal_myocardium, 2=edema, 3=scar.
          - CINE: contrasts bg (0) vs myocardial_ring ({1,2,3} -> 1).
          - PSIR (LGE) & T2w: ignores bg (0), contrasts normal_myo (1) vs edema (2) vs scar (3).
        """
        protos = mod_aux["class_protos"].float()  # (3, 4, D)
        device = target.device
        modality_specs = (
            ("emb_cine", "cine"),
            ("emb_psir", "intra_ring"),
            ("emb_t2w", "intra_ring"),
        )
        total_mod_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        for key, mode in modality_specs:
            embeds = mod_aux[key]
            mod_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            for s_idx, (feat, n_view) in enumerate(zip(embeds, n_view_list)):
                b, d, h, w = feat.shape
                t_ds = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()
                if mode == "cine":
                    t_mod = (t_ds > 0).long()
                    allowed = (0, 1)
                    proto_vecs = protos[s_idx, :2]
                    proto_lbls = torch.tensor((0, 1), device=device, dtype=torch.long)
                else:
                    t_mod = t_ds
                    allowed = (1, 2, 3)
                    proto_vecs = protos[s_idx, 1:4]
                    proto_lbls = torch.tensor((1, 2, 3), device=device, dtype=torch.long)

                feat_flat = feat.float().permute(0, 2, 3, 1).reshape(b, h * w, d)
                t_flat = t_mod.reshape(b, h * w)
                anc_feats = []
                anc_lbls = []
                for bi in range(b):
                    tb = t_flat[bi]
                    fb = feat_flat[bi]
                    for cls_id in allowed:
                        idx = torch.nonzero(tb == cls_id, as_tuple=False).squeeze(1)
                        cnt = idx.numel()
                        if cnt == 0:
                            continue
                        if cnt > n_view:
                            perm = torch.randperm(cnt, device=device)[:n_view]
                            idx = idx[perm]
                        anc_feats.append(fb[idx])
                        anc_lbls.append(torch.full((idx.numel(),), cls_id, device=device, dtype=torch.long))
                if not anc_feats:
                    mod_loss = mod_loss + feat.float().sum() * 0.0 + proto_vecs.sum() * 0.0
                    continue
                all_feats = torch.cat(anc_feats + [proto_vecs], dim=0)
                all_feats = F.normalize(all_feats, p=2, dim=1, eps=1e-6)
                all_lbls = torch.cat(anc_lbls + [proto_lbls], dim=0)
                mod_loss = mod_loss + self._supcon_loss(all_feats, all_lbls, temperature=0.07)
            total_mod_loss = total_mod_loss + mod_loss / len(n_view_list)
        return total_mod_loss / len(modality_specs)

    def forward(self, output, target):
        mod_aux = None
        if isinstance(output, dict):
            logits = output["logits"]
            mod_aux = output.get("mod_aux")
        else:
            logits = output
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

        # 6. Clean Hierarchical Composite Batch-Dice (MyoRing, AAR/Edema-Incl, Scar),
        #    Per-Slice Positive-Masked Hierarchical Tversky/Dice,
        #    and Distance-Transform Hausdorff Surrogate Loss
        if (
            self.hier_dice_weight > 0.0
            or self.slice_hier_weight > 0.0
            or self.boundary_dt_weight > 0.0
            or self.norm_myo_dt_weight > 0.0
            or self.intra_myo_far_fp_weight > 0.0
            or self.empty_slice_fp_weight > 0.0
        ):
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
            hier_rw_vals = self.hier_region_weights if self.hier_region_weights is not None else (0.30, 0.35, 0.35)
            hier_rw = torch.tensor(hier_rw_vals, device=logits.device, dtype=torch.float32)
            hier_rw = hier_rw / hier_rw.sum()
            bdt_rw_default = (0.30, 0.40, 0.30) if self.boundary_dt_mode == "hybrid_karimi_farfp" else (0.30, 0.35, 0.35)
            bdt_rw_vals = self.boundary_dt_region_weights if self.boundary_dt_region_weights is not None else bdt_rw_default
            bdt_rw = torch.tensor(bdt_rw_vals, device=logits.device, dtype=torch.float32)
            bdt_rw = bdt_rw / bdt_rw.sum()

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
            hier_dice_loss = (per_reg_h * hier_rw).sum()
        else:
            hier_dice_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        if self.slice_hier_weight > 0.0:
            # Per-slice positive-masked hierarchical Tversky/Dice: ensures smaller/fainter
            # apical and basal positive slices (which are 15-46x smaller than mid-ventricular
            # slices in the batch sum) receive equal slice-level gradient.
            betas = torch.tensor(
                (self.ring_tversky_beta, self.aar_tversky_beta, 0.5),
                device=logits.device,
                dtype=torch.float32,
            ).view(1, 3)
            alphas = 1.0 - betas
            tp_s = (p_pos * y_pos).sum(dim=(2, 3))
            fp_s = (p_pos * (1.0 - y_pos)).sum(dim=(2, 3))
            fn_s = (p_neg * y_pos).sum(dim=(2, 3))
            slice_scores = (tp_s + 1e-5) / (tp_s + alphas * fp_s + betas * fn_s + 1e-5)
            slice_reg_losses = 1.0 - slice_scores  # (B, 3)
            has_fg_s = (y_pos.sum(dim=(2, 3)) > 0).float()  # (B, 3)
            pos_counts = has_fg_s.sum(dim=0).clamp_min(1.0)
            per_reg_slice_h = (slice_reg_losses * has_fg_s).sum(dim=0) / pos_counts
            slice_hier_loss = (per_reg_slice_h * hier_rw).sum()
        else:
            slice_hier_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        if self.boundary_dt_weight > 0.0:
            y_neg = 1.0 - y_pos
            fg_denom = y_pos.sum(dim=(0, 2, 3)) + p_pos.detach().sum(dim=(0, 2, 3)) + 1.0
            myo_boost = None
            if self.boundary_dt_myo_fp_alpha > 0.0:
                in_norm_myo = (target == 1).float().unsqueeze(1)
                myo_fp_scale = 1.0 + self.boundary_dt_myo_fp_alpha * in_norm_myo
                myo_boost = torch.cat((torch.ones_like(in_norm_myo), myo_fp_scale, myo_fp_scale), dim=1)
            if self.boundary_dt_scales is not None:
                scale_3 = torch.tensor(self.boundary_dt_scales, device=logits.device, dtype=torch.float32).view(1, 3, 1, 1)
                scale_6 = torch.cat((scale_3, scale_3), dim=1)
            else:
                scale_3 = self.boundary_dt_scale
                scale_6 = self.boundary_dt_scale
            if self.boundary_dt_empty_vals is not None:
                empty_3 = torch.tensor(self.boundary_dt_empty_vals, device=logits.device, dtype=torch.float32).view(1, 3, 1, 1)
                empty_pr_3 = torch.full_like(empty_3, self.boundary_dt_empty_val)
                empty_6 = torch.cat((empty_3, empty_pr_3), dim=1)
            else:
                empty_3 = self.boundary_dt_empty_val
                empty_6 = self.boundary_dt_empty_val
            if self.boundary_dt_mode == "fp_bce":
                w_gt = _octagonal_outside_dt(
                    y_pos,
                    max_steps=self.boundary_dt_max_steps,
                    d_scale=scale_3,
                    power=self.boundary_dt_power,
                    empty_val=empty_3,
                    margin=self.boundary_dt_margin,
                    max_norm=self.boundary_dt_max_norm,
                )
                if myo_boost is not None:
                    w_gt = w_gt * myo_boost
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

                n_gt = d_gt / scale_3
                n_pr = d_pr / scale_3
                if self.boundary_dt_max_norm is not None:
                    n_gt = n_gt.clamp(max=self.boundary_dt_max_norm)
                    n_pr = n_pr.clamp(max=self.boundary_dt_max_norm)
                w_gt = torch.where(
                    h_gt,
                    n_gt.pow(self.boundary_dt_power),
                    torch.full_like(d_gt, 1.0),
                )
                w_pr = torch.where(
                    h_pr,
                    n_pr.pow(self.boundary_dt_power),
                    torch.full_like(d_pr, 1.0),
                )
                # Dead-zone margin: half-margin on thin MyoRing, full margin on AAR/Scar
                # so near-boundary Dice calibration is untouched while distant FPs get crushed.
                margins = torch.tensor(
                    (0.5 * self.boundary_dt_margin, self.boundary_dt_margin, self.boundary_dt_margin),
                    device=logits.device,
                    dtype=torch.float32,
                ).view(1, 3, 1, 1)
                n_far = (d_gt - margins).clamp_min(0.0) / scale_3
                if self.boundary_dt_max_norm is not None:
                    n_far = n_far.clamp(max=self.boundary_dt_max_norm)
                empty_map = (
                    empty_3.expand_as(d_gt) if isinstance(empty_3, torch.Tensor) else torch.full_like(d_gt, empty_3)
                )
                w_far = torch.where(
                    h_gt,
                    n_far.pow(self.boundary_dt_power),
                    empty_map,
                )
                if myo_boost is not None:
                    w_gt = w_gt * myo_boost
                    w_far = w_far * myo_boost
                karimi_reg = (y_neg * p_pos.square() * w_gt + y_pos * p_neg.square() * w_pr).sum(dim=(0, 2, 3)) / fg_denom
                fp_err = (-torch.log(p_neg.clamp(min=1e-6))).clamp(max=10.0)
                far_fp_reg = (w_far * y_neg * fp_err).sum(dim=(0, 2, 3)) / fg_denom
                per_reg = karimi_reg + 0.30 * far_fp_reg
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
                    d_scale=scale_6,
                    power=self.boundary_dt_power,
                    empty_val=empty_6,
                    margin=self.boundary_dt_margin,
                    max_norm=self.boundary_dt_max_norm,
                )
                w_gt, w_pr = both_dt[:, :3], both_dt[:, 3:]
                if myo_boost is not None:
                    w_gt = w_gt * myo_boost
                if self.boundary_dt_mode == "sym_bce":
                    fp_err = (-torch.log(p_neg.clamp(min=1e-6))).clamp(max=10.0)
                    fn_err = (-torch.log(p_pos.clamp(min=1e-6))).clamp(max=10.0)
                    per_reg = (w_gt * y_neg * fp_err + w_pr * y_pos * fn_err).sum(dim=(0, 2, 3)) / fg_denom
                else:  # karimi_mse, karimi_qlog, or karimi_fp_qlog
                    if self.boundary_dt_mode == "karimi_qlog":
                        fp_term = p_pos * (-torch.log(p_neg.clamp(min=1e-4)))
                        fn_term = p_neg * (-torch.log(p_pos.clamp(min=1e-4)))
                    elif self.boundary_dt_mode == "karimi_fp_qlog":
                        fp_term = p_pos * (-torch.log(p_neg.clamp(min=1e-4)))
                        fn_term = p_neg.square()
                    else:
                        fp_term = p_pos.square()
                        fn_term = p_neg.square()
                    err_map = y_neg * fp_term * w_gt + y_pos * fn_term * w_pr
                    batch_reg = err_map.sum(dim=(0, 2, 3)) / fg_denom
                    if self.boundary_dt_slice_weight > 0.0:
                        slice_floors = torch.tensor((400.0, 200.0, 100.0), device=logits.device, dtype=torch.float32).view(1, 3)
                        slice_denom = y_pos.sum(dim=(2, 3)) + p_pos.detach().sum(dim=(2, 3)) + slice_floors
                        slice_reg = (err_map.sum(dim=(2, 3)) / slice_denom).mean(dim=0)
                        sw = self.boundary_dt_slice_weight
                        per_reg = (1.0 - sw) * batch_reg + sw * slice_reg
                    else:
                        per_reg = batch_reg
            boundary_dt_loss = (per_reg * bdt_rw).sum()
        else:
            boundary_dt_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        if self.norm_myo_dt_weight > 0.0:
            y_myo = (target == 1).float().unsqueeze(1)
            p_myo = probs[:, 1:2]
            p_not_myo = probs[:, 0:1] + probs[:, 2:3] + probs[:, 3:4]
            inter_myo = 2.0 * (p_myo * y_myo).sum() + 1e-5
            denom_myo = p_myo.square().sum() + y_myo.sum() + 1e-5
            myo_dice_loss = 1.0 - inter_myo / denom_myo

            pred_myo = (logits.detach().argmax(1) == 1).float().unsqueeze(1)
            myo_dt = _octagonal_outside_dt(
                torch.cat((y_myo, pred_myo), dim=1),
                max_steps=self.boundary_dt_max_steps,
                d_scale=12.0,
                power=self.boundary_dt_power,
                empty_val=self.boundary_dt_empty_val,
            )
            w_myo_gt, w_myo_pr = myo_dt[:, :1], myo_dt[:, 1:]
            myo_denom = y_myo.sum() + p_myo.detach().sum() + 1.0
            myo_err = (1.0 - y_myo) * p_myo.square() * w_myo_gt + y_myo * p_not_myo.square() * w_myo_pr
            myo_karimi_loss = myo_err.sum() / myo_denom
            norm_myo_dt_loss = 0.5 * myo_dice_loss + 0.5 * myo_karimi_loss
        else:
            norm_myo_dt_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        if self.intra_myo_far_fp_weight > 0.0:
            y_myo_m = (target == 1).float().unsqueeze(1)
            y_edema_m = (target == 2).float().unsqueeze(1)
            y_aar_m = ((target == 2) | (target == 3)).float().unsqueeze(1)
            y_scar_m = (target == 3).float().unsqueeze(1)
            pred_cls_m = logits.detach().argmax(1)
            pred_myo_m = (pred_cls_m == 1).float().unsqueeze(1)

            d_64, has_64 = _octagonal_outside_dist(torch.cat((y_aar_m, y_scar_m), dim=1), max_steps=64)
            d_aar_64, has_aar_m = d_64[:, 0:1], has_64[:, 0:1]
            d_scar_64, has_scar_m = d_64[:, 1:2], has_64[:, 1:2]
            d_pr_myo, has_pr_myo = _octagonal_outside_dist(pred_myo_m, max_steps=32)

            far_gate = torch.where(has_aar_m, (d_aar_64 > 8.0).float(), torch.ones_like(d_aar_64))
            w_far_aar = torch.where(
                has_aar_m,
                ((d_aar_64 - 8.0).clamp_min(0.0) / 20.0).square(),
                torch.full_like(d_aar_64, 1.5),
            )
            w_hole_myo = torch.where(
                has_pr_myo,
                ((d_pr_myo - 2.0).clamp_min(0.0) / 12.0).square(),
                torch.ones_like(d_pr_myo),
            ) * far_gate

            p_myo_m = probs[:, 1:2]
            p_edema_m = probs[:, 2:3]
            p_scar_m = probs[:, 3:4]
            p_aar_m = p_edema_m + p_scar_m
            p_ring_m = (p_myo_m + p_aar_m).clamp_min(1e-4)
            p_ring_det = (p_myo_m + p_aar_m).detach()

            q_aar_in_ring = (p_aar_m / p_ring_m).clamp(0.0, 1.0 - 1e-4)
            q_myo_in_ring = (p_myo_m / p_ring_m).clamp_min(1e-4)
            qlog_intra_aar = p_ring_det * q_aar_in_ring * (-torch.log(q_myo_in_ring))
            aar_far_loss = (y_myo_m * (w_far_aar + w_hole_myo) * qlog_intra_aar).sum() / (y_myo_m.sum() + 100.0)

            p_aar_safe = p_aar_m.clamp_min(1e-4)
            p_aar_det = p_aar_m.detach()
            q_scar_in_aar = (p_scar_m / p_aar_safe).clamp(0.0, 1.0 - 1e-4)
            q_edema_in_aar = (p_edema_m / p_aar_safe).clamp_min(1e-4)
            qlog_scar_in_ede = p_aar_det * q_scar_in_aar * (-torch.log(q_edema_in_aar))

            q_scar_in_ring = (p_scar_m / p_ring_m).clamp(0.0, 1.0 - 1e-4)
            q_non_scar_in_ring = ((p_myo_m + p_edema_m) / p_ring_m).clamp_min(1e-4)
            qlog_scar_in_myo = p_ring_det * q_scar_in_ring * (-torch.log(q_non_scar_in_ring))

            w_far_scar_ede = torch.where(
                has_scar_m,
                ((d_scar_64 - 4.0).clamp_min(0.0) / 14.0).square(),
                torch.full_like(d_scar_64, 2.5),
            )
            w_far_scar_myo = torch.where(
                has_scar_m,
                ((d_scar_64 - 8.0).clamp_min(0.0) / 16.0).square(),
                torch.full_like(d_scar_64, 2.5),
            )
            scar_pixel_err = y_edema_m * w_far_scar_ede * qlog_scar_in_ede + y_myo_m * w_far_scar_myo * qlog_scar_in_myo
            y_non_scar_ring = y_myo_m + y_edema_m
            scar_far_batch = scar_pixel_err.sum() / (y_non_scar_ring.sum() + 100.0)

            empty_scar_s = (~has_scar_m).float()
            empty_scar_err = (
                (y_edema_m * qlog_scar_in_ede + y_myo_m * qlog_scar_in_myo) * empty_scar_s
            ).sum(dim=(1, 2, 3)) / (y_non_scar_ring.sum(dim=(1, 2, 3)) + 100.0)
            empty_scar_slice_loss = empty_scar_err.sum() / (empty_scar_s.sum() + 1.0)

            intra_myo_far_fp_loss = aar_far_loss + 0.35 * scar_far_batch + 0.50 * empty_scar_slice_loss
        else:
            intra_myo_far_fp_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        if self.empty_slice_fp_weight > 0.0:
            has_fg_e = (y_pos.sum(dim=(2, 3)) > 0)
            is_empty_e = (~has_fg_e[:, 1:]).float()
            ring_boost_e = 1.0 + 2.0 * y_pos[:, 0:1]
            fp_qlog_e = p_pos[:, 1:] * (-torch.log(p_neg[:, 1:].clamp(min=1e-4)))
            slice_ring_denom = y_pos[:, 0:1].sum(dim=(2, 3)) + p_pos[:, 1:].detach().sum(dim=(2, 3)) + 200.0
            slice_empty_err = (ring_boost_e * fp_qlog_e).sum(dim=(2, 3)) / slice_ring_denom
            per_reg_empty = (slice_empty_err * is_empty_e).sum(dim=0) / (is_empty_e.sum(dim=0) + 1.0)
            empty_slice_fp_loss = 0.35 * per_reg_empty[0] + 0.65 * per_reg_empty[1]
        else:
            empty_slice_fp_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        if self.mod_aux_weight > 0.0 and mod_aux is not None:
            t_ring = (target > 0).long()
            # PSIR/LGE: 0=bg, 1=non-scar myo (1 or 2), 2=scar (3)
            t_psir = torch.where(target == 3, 2, (target > 0).long())
            # T2w: 0=bg, 1=normal_myo (1), 2=AAR/edema_inclusive (2 or 3)
            t_t2w = torch.where((target == 2) | (target == 3), 2, (target == 1).long())
            loss_aux_cine = self._aux_ce_dice(mod_aux["aux_cine"], t_ring, (0.5, 1.0))
            loss_aux_psir = self._aux_ce_dice(mod_aux["aux_psir"], t_psir, (0.5, 1.0, 1.15))
            loss_aux_t2w = self._aux_ce_dice(mod_aux["aux_t2w"], t_t2w, (0.5, 1.1, 1.1))
            mod_aux_loss = (loss_aux_cine + loss_aux_psir + loss_aux_t2w) / 3.0
        else:
            mod_aux_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        if self.mod_contrast_weight > 0.0 and mod_aux is not None:
            mod_contrast_loss = self._compute_modality_contrastive(mod_aux, target)
        else:
            mod_contrast_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        total_loss = (
            self.ce_weight * ce
            + self.dice_weight * dice
            + self.aar_weight * aar_loss
            + self.scar_weight * scar_loss
            + self.anat_contrast_weight * anat_contrast_loss
            + self.hier_dice_weight * hier_dice_loss
            + self.slice_hier_weight * slice_hier_loss
            + self.boundary_dt_weight * boundary_dt_loss
            + self.norm_myo_dt_weight * norm_myo_dt_loss
            + self.intra_myo_far_fp_weight * intra_myo_far_fp_loss
            + self.empty_slice_fp_weight * empty_slice_fp_loss
            + self.mod_aux_weight * mod_aux_loss
            + self.mod_contrast_weight * mod_contrast_loss
        )
        result = {
            "loss": total_loss,
            "ce": ce,
            "dice_loss": dice,
            "aar_loss": aar_loss,
            "scar_loss": scar_loss,
            "anat_contrast_loss": anat_contrast_loss,
        }
        if self.hier_dice_weight > 0.0 or self.slice_hier_weight > 0.0 or self.boundary_dt_weight > 0.0:
            result["hier_dice_loss"] = hier_dice_loss + slice_hier_loss
            result["boundary_dt_loss"] = boundary_dt_loss
        if self.norm_myo_dt_weight > 0.0:
            result["norm_myo_dt_loss"] = norm_myo_dt_loss
        if self.intra_myo_far_fp_weight > 0.0:
            result["intra_myo_far_fp_loss"] = intra_myo_far_fp_loss
        if self.empty_slice_fp_weight > 0.0:
            result["empty_slice_fp_loss"] = empty_slice_fp_loss
        if self.mod_aux_weight > 0.0:
            result["mod_aux_loss"] = mod_aux_loss
        if self.mod_contrast_weight > 0.0:
            result["mod_contrast_loss"] = mod_contrast_loss
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
