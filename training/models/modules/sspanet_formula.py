"""SCAR's explicit interpretation of SSPANet equations (17)-(23).

Reference: Hasan et al., BSPC 112 (2026), 108636, section 3.5.
The paper's equations, diagrams and released demo disagree. This version follows
the sequential equations, not the author's parallel residual demo. See
docs/sspanet_paper_formula.md for the choices needed to make the equations exact.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


PAPER_FORMULA_GATE = "paper_formula_v1"


class _PooledBatchNorm2d(nn.BatchNorm2d):
    """Use stored statistics when a pooled training batch has only one value."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.numel() // x.shape[1] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, False, self.momentum, self.eps)
        return super().forward(x)


def _stats_input(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


class PaperFormulaSSPANet(nn.Module):
    """Channel attention followed by one-channel strip/style spatial attention.

    Mc = sigmoid(BN(Conv1x1(cat[mean_HW(X), max_HW(X)])))
    Xp = Mc * X
    Sh = mean_H(Xp), Sv = mean_W(Xp)
    strip = Conv1x1(cat[broadcast(Sh), broadcast(Sv)])  # 2C -> 1
    style = Conv1x1(cat[mean_C(Xp), sqrt(var_C(Xp) + eps)])  # 2 -> 1
    Ms = sigmoid(Conv1x1(strip + style))
    Y = Ms * Xp

    Strip projection is evaluated before broadcast using slices of the same
    kernel, preserving the equation and applying the bias exactly once. There
    is one sigmoid per attention map and no residual, tanh or reduction ratio.
    """

    def __init__(self, in_channels: int = 1024):
        super().__init__()
        if type(in_channels) is not int or in_channels < 1:
            raise ValueError("in_channels must be a positive integer")
        self.in_channels = in_channels
        # Fixed as part of paper_formula_v1, including population variance.
        self.eps = 1e-6
        self.channel_proj = nn.Conv2d(2 * in_channels, in_channels, 1, bias=False)
        self.channel_norm = _PooledBatchNorm2d(in_channels)
        self.strip_proj = nn.Conv2d(2 * in_channels, 1, 1, bias=True)
        self.style_proj = nn.Conv2d(2, 1, 1, bias=True)
        self.spatial_proj = nn.Conv2d(1, 1, 1, bias=True)

    def style_statistics(self, x: torch.Tensor) -> torch.Tensor:
        """Equation (22): mean/std across C at each pixel, not across H/W."""
        values = _stats_input(x)
        mean = values.mean(dim=1, keepdim=True)
        variance = (values - mean).square().mean(dim=1, keepdim=True)
        return torch.cat((mean, (variance + self.eps).sqrt()), dim=1).to(x.dtype)

    def strip_logits(self, x: torch.Tensor) -> torch.Tensor:
        values = _stats_input(x)
        sh = values.mean(dim=2, keepdim=True).to(x.dtype)  # B,C,1,W
        sv = values.mean(dim=3, keepdim=True).to(x.dtype)  # B,C,H,1
        weight = self.strip_proj.weight
        horizontal = F.conv2d(sh, weight[:, :self.in_channels], self.strip_proj.bias)
        vertical = F.conv2d(sv, weight[:, self.in_channels:])
        return horizontal + vertical

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected (B,{self.in_channels},H,W), got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("SSPANet input must be floating point")
        values = _stats_input(x)
        descriptor = torch.cat((values.mean(dim=(2, 3), keepdim=True),
                                values.amax(dim=(2, 3), keepdim=True)), dim=1).to(x.dtype)
        mc = torch.sigmoid(self.channel_norm(self.channel_proj(descriptor)))
        refined = x * mc
        style = self.style_proj(self.style_statistics(refined))
        ms = torch.sigmoid(self.spatial_proj(self.strip_logits(refined) + style))
        output = refined * ms
        if return_attention:
            return output, mc, ms
        return output
