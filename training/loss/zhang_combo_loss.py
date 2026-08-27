"""Weighted Categorical Cross-Entropy + Soft Multiclass Dice Loss (ComboLoss) for Zhang (2021).

Reference:
    - Taghanaki, S.A. et al. (2019). Combo loss: Handling input and output imbalance
      in multi-organ segmentation. Comput Med Imaging Graph 75, 24-33.
    - Lalande, A. et al. (2022). "for the scar segmentation, the categorical cross entropy
      loss was weighted (Zhang...) while the multi-class Dice loss was not weighted...
      combination termed Comboloss was also practiced..."
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def multiclass_soft_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    smooth: float = 1e-6,
    include_background: bool = False,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes soft multi-class Dice loss over spatial dimensions.

    Args:
        logits: Shape (B, C, D, H, W) or (B, C, H, W)
        labels: Shape (B, D, H, W) or (B, H, W) with integer class IDs [0..C-1]
        num_classes: Total number of classes
        smooth: Laplace smoothing epsilon
        include_background: Whether to include class 0 (background)
        class_weights: Optional per-class weight tensor of shape (num_classes,)
    """
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(labels.long(), num_classes=num_classes).movedim(-1, 1).float()

    start_idx = 0 if include_background else 1
    probs_fg = probs[:, start_idx:]
    one_hot_fg = one_hot[:, start_idx:]

    dims = tuple(i for i in range(probs.ndim) if i != 1)
    intersection = torch.sum(probs_fg * one_hot_fg, dim=dims)
    cardinality = torch.sum(probs_fg + one_hot_fg, dim=dims)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    dice_loss = 1.0 - dice  # (B, num_fg_classes)

    if class_weights is not None:
        weights = class_weights[start_idx:].to(logits.device)
        dice_loss = dice_loss * weights

    return dice_loss.mean()


class ZhangComboLoss(nn.Module):
    """Zhang (2021) Weighted Cross-Entropy + Multiclass Dice Loss."""

    def __init__(
        self,
        num_classes: int = 5,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[Union[List[float], torch.Tensor]] = None,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights)
        else:
            # Default weights giving higher emphasis on rare pathologies (MI and PMO/Scar)
            # Default 5 classes: [0=bg (0.5), 1=lv_cavity (1.0), 2=lv_myo (1.5), 3=scar (3.0), 4=rv_cavity (1.0)]
            default_w = torch.tensor([0.5, 1.0, 1.5, 3.0, 1.0], dtype=torch.float32)
            if num_classes != 5:
                default_w = torch.ones(num_classes, dtype=torch.float32)
            self.register_buffer("class_weights", default_w)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate weighted CE + Dice loss."""
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None

        ce_loss = F.cross_entropy(logits, labels.long(), weight=w)
        dice_loss = multiclass_soft_dice_loss(
            logits,
            labels,
            num_classes=self.num_classes,
            smooth=self.smooth,
            include_background=False,
        )

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class ZhangCascadedLoss(nn.Module):
    """Joint Multi-Stage Loss for Cascaded 2D-3D Framework.

    Loss = Loss_fine(stage2_fine_logits, labels) + coarse_loss_weight * Loss_coarse(stage1_coarse_logits, labels)
    """

    def __init__(
        self,
        num_classes: int = 5,
        coarse_loss_weight: float = 0.5,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[Union[List[float], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.coarse_loss_weight = coarse_loss_weight
        self.combo_loss = ZhangComboLoss(
            num_classes=num_classes,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            class_weights=class_weights,
        )

    def forward(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return self.combo_loss(outputs, labels)

        fine_logits = outputs["fine_logits"]
        coarse_logits = outputs.get("coarse_logits")

        loss_fine = self.combo_loss(fine_logits, labels)

        if coarse_logits is not None and self.coarse_loss_weight > 0:
            loss_coarse = self.combo_loss(coarse_logits, labels)
            return loss_fine + self.coarse_loss_weight * loss_coarse

        return loss_fine
