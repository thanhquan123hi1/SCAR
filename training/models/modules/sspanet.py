"""Optimized Strip-Style Pooling Attention Network (OptimizedSSPANet) and legacy SSPA modules.

Mathematically sound reconstruction resolving:
1. Finding D-01: Gating dynamic range collapse eliminated via unified single-sigmoid on additive logits
   spanning the full range [0.0, 1.0].
2. Parameter explosion: 22.04M -> 1.38M for 3 bottleneck encoders (93.72% reduction) using reduction ratio
   r=16 and depthwise separable convolutions.
3. Compute efficiency: linear projections on 1D strips before spatial broadcasting.
4. Restored Style Pooling: extracts 1st-order (spatial mean mu) and 2nd-order (spatial standard deviation
   sigma = sqrt(Var + eps)) microtexture statistics for myocardial scar and edema tissue discrimination.

Literature References:
- Md Jahid Hasan et al., "Enhancing brain tumor classification with a novel attention based explainable
  deep learning framework", Biomedical Signal Processing and Control, Vol. 112, 108636 (2026).
  DOI: 10.1016/j.bspc.2025.108636.
- Style-based Recalibration Module (SRM), Lee et al., ICCV 2019.
- Strip Pooling: Rethinking Spatial Pooling for Scene Parsing (SPNet), Hou et al., CVPR 2020.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


OPTIMIZED_SSPA_GATE = "optimized"
CENTERED_SSPA_GATE = "centered_tanh_v2"
LEGACY_SSPA_GATE = "legacy_v1"
DEFAULT_SSPA_GATE = "centered_tanh_v2"
SSPA_GATE_MODES = (OPTIMIZED_SSPA_GATE, CENTERED_SSPA_GATE, LEGACY_SSPA_GATE)


def _statistics_input(x: torch.Tensor) -> torch.Tensor:
    """Accumulate squares and variance in FP32 when using AMP."""
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def strip_rms(x: torch.Tensor, dim: int, epsilon: float = 1e-6) -> torch.Tensor:
    """Author's uncentered RMS statistic, retaining the pooled dimension."""
    return (_statistics_input(x).square().mean(dim=dim, keepdim=True) + epsilon).sqrt()


class _SafeBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d that safely handles single-instance (1x1) spatial boundary conditions during training."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and (input.numel() // input.size(1) <= 1):
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(input)


# ==============================================================================
# Legacy SSPANet Modules (Retained for Backward Checkpoint & Model Compatibility)
# ==============================================================================

class SSPA_BasicConv(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
        relu: bool = True,
        bn: bool = True,
        bias: bool = False,
    ):
        layers = [
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        ]
        if bn:
            layers.append(nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01))
        if relu:
            layers.append(nn.ReLU())
        super().__init__(*layers)


class SSPA_ZPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reduce C, not H/W: (B,C,H,W) -> (B,2,H,W).
        return torch.cat(
            (x.max(dim=1, keepdim=True).values, x.mean(dim=1, keepdim=True)),
            dim=1,
        )


class SSPA_ChannelAttention(nn.Module):
    """Author's legacy module: channel pooling produces a spatial gate."""

    def __init__(self):
        super().__init__()
        self.compress = SSPA_ZPool()
        self.conv = SSPA_BasicConv(2, 1, 7, padding=3, relu=False)

    def gate_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Signed gate logits, before sigmoid or feature multiplication."""
        return self.conv(self.compress(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gate_logits(x))


class SSPA_SpatialAttention(nn.Module):
    """Legacy spatial attention module with dense C x C strip convolutions."""

    def __init__(self, inplanes: int, outplanes: int | None = None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        outplanes = inplanes if outplanes is None else outplanes
        if inplanes != outplanes:
            raise ValueError("SSPA multiplicative attention must preserve channels")
        self.conv1 = nn.Conv2d(inplanes, outplanes, (3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = nn.Conv2d(inplanes, outplanes, (1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(outplanes)
        self.conv3 = nn.Conv2d(outplanes, outplanes, 1)

    def gate_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Signed strip-attention logits with the original projection weights."""
        rms_h = strip_rms(x, dim=3).to(x.dtype)
        rms_w = strip_rms(x, dim=2).to(x.dtype)
        strip_h = self.bn1(self.conv1(rms_h))
        strip_w = self.bn2(self.conv2(rms_w))
        projected_h = F.conv2d(strip_h, self.conv3.weight, self.conv3.bias)
        projected_w = F.conv2d(strip_w, self.conv3.weight)
        return projected_h + projected_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gate_logits(x))


# ==============================================================================
# Standardized & Optimized SSPANet Modules (Milestone 1)
# ==============================================================================

class SSPA_ZPoolChannelAttention(nn.Module):
    """Component A: Channel attention via multi-pooling (Z-Pool) and shared bottleneck MLP.

    Computes channel-wise attention logits M_c in R^(B x C x 1 x 1) using both
    average pooling and max pooling compressed via a shared MLP with reduction ratio r.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(in_channels // reduction, 16)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes raw channel attention logits M_c in R^(B x C x 1 x 1)."""
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        return self.mlp(avg_pool) + self.mlp(max_pool)


class SSPA_DirectionalStripPooling(nn.Module):
    """Component B: Directional strip pooling via depthwise separable convolutions and 1D projections.

    Captures long-range horizontal and vertical anatomical context using:
    - Strip pooling: (B, C, H, 1) and (B, C, 1, W)
    - 1D Bottleneck channel reduction (C -> C/r)
    - Depthwise convolutions (3x1 and 1x3)
    - 1D linear projection (C/r -> C) before 2D broadcast addition
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(in_channels // reduction, 16)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            _SafeBatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                (3, 1),
                padding=(1, 0),
                groups=mid_channels,
                bias=False,
            ),
            _SafeBatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                (1, 3),
                padding=(0, 1),
                groups=mid_channels,
                bias=False,
            ),
            _SafeBatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.expand = nn.Conv2d(mid_channels, in_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes raw directional strip attention logits M_strip in R^(B x C x H x W)."""
        b, c, h, w = x.shape
        strip_h = F.adaptive_avg_pool2d(x, (h, 1))
        strip_w = F.adaptive_avg_pool2d(x, (1, w))

        red_h = self.reduce(strip_h)
        red_w = self.reduce(strip_w)

        feat_h = self.conv_h(red_h)
        feat_w = self.conv_w(red_w)

        # 1D linear projection before broadcast: mathematically identical to expanding after broadcast,
        # The expand projection uses H+W positions instead of H*W (8x at 16x16).
        exp_h = F.conv2d(feat_h, self.expand.weight, self.expand.bias)
        exp_w = F.conv2d(feat_w, self.expand.weight)
        return exp_h + exp_w


class SSPA_StylePooling(nn.Module):
    """Component C: Style pooling capturing 1st-order (mean) and 2nd-order (std dev) microtexture statistics.

    Clinical and Biophysical Relevance:
    - Normal myocardium exhibits uniform cellular structure (low spatial variance sigma).
    - Myocardial edema presents capillary leakage and diffuse hyperintensity (high mean mu, moderate sigma).
    - Myocardial scar exhibits patchy replacement fibrosis and contrast rim boundaries (high mu, high sigma).
    """

    def __init__(self, in_channels: int, reduction: int = 16, eps: float = 1e-6):
        super().__init__()
        mid_channels = max(in_channels // reduction, 16)
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes raw style recalibration logits M_style in R^(B x C x 1 x 1)."""
        x_fp = _statistics_input(x)
        mean = x_fp.mean(dim=(2, 3), keepdim=True)
        var = ((x_fp - mean) ** 2).mean(dim=(2, 3), keepdim=True)
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var + self.eps)
        style_stats = torch.cat([mean, std], dim=1).to(x.dtype)
        return self.mlp(style_stats)


class OptimizedSSPANet(nn.Module):
    """Standardized, unified Strip-Style Pooling Attention Network.

    Combines Z-Pool Channel Attention, Depthwise Directional Strip Pooling,
    and Restored Style Pooling before a single sigmoid activation:
        gate = sigmoid(M_c + M_strip + M_style) in [0.0, 1.0]

    Modes:
    - 'residual': output = x + x * gate (for bottleneck feature refinement, [1.0x, 2.0x])
    - 'attention_gate': output = x * gate (for skip-connection noise suppression, [0.0x, 1.0x])
    """

    def __init__(
        self,
        in_channels: int = 1024,
        reduction_ratio: int = 16,
        reduction: int | None = None,
        mode: str = "residual",
        is_skip_gate: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        r = reduction if reduction is not None else reduction_ratio
        if is_skip_gate:
            mode = "attention_gate"
        if mode not in ("residual", "attention_gate"):
            raise ValueError(f"Unknown mode {mode!r}; expected 'residual' or 'attention_gate'")
        self.in_channels = in_channels
        self.reduction_ratio = r
        self.mode = mode
        self.is_skip_gate = is_skip_gate

        self.channel_attn = SSPA_ZPoolChannelAttention(in_channels, reduction=r)
        self.strip_pooling = SSPA_DirectionalStripPooling(in_channels, reduction=r)
        self.style_pooling = SSPA_StylePooling(in_channels, reduction=r, eps=eps)

    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute unified additive projection logits before activation."""
        m_c = self.channel_attn(x)
        m_strip = self.strip_pooling(x)
        m_style = self.style_pooling(x)
        return m_c + m_strip + m_style

    def forward(
        self,
        x: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits = self.compute_logits(x)
        gate = torch.sigmoid(logits)
        if self.mode == "attention_gate":
            out = x * gate
        else:
            out = x + x * gate
        if return_gate:
            return out, gate
        return out


class SSPANet_Block(nn.Module):
    """SSPANet Block supporting optimized, centered_tanh_v2, and legacy_v1 gating.

    Default gate_mode is centered_tanh_v2; optimized must be selected explicitly.
    Legacy modes ('centered_tanh_v2', 'legacy_v1') are retained for full backward compatibility
    with existing pretrained checkpoints and legacy regression tests.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        gate_mode: str = DEFAULT_SSPA_GATE,
        reduction_ratio: int = 16,
    ):
        super().__init__()
        if gate_mode not in SSPA_GATE_MODES:
            raise ValueError(f"Unknown SSPANet gate {gate_mode!r}; expected {SSPA_GATE_MODES}")
        self.gate_mode = gate_mode
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        if gate_mode == OPTIMIZED_SSPA_GATE:
            self.optimized = OptimizedSSPANet(
                in_channels=in_channels,
                reduction_ratio=reduction_ratio,
                mode="residual",
            )
            self.channel_attn = self.optimized.channel_attn
            self.strip_pooling = self.optimized.strip_pooling
            self.style_pooling = self.optimized.style_pooling
        else:
            self.channel = SSPA_ChannelAttention()
            self.spatial = SSPA_SpatialAttention(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == OPTIMIZED_SSPA_GATE:
            return self.optimized(x)
        if self.gate_mode == LEGACY_SSPA_GATE:
            return x + x * torch.sigmoid(self.channel(x) + self.spatial(x))
        logits = self.channel.gate_logits(x) + self.spatial.gate_logits(x)
        return x * (1.0 + 0.5 * torch.tanh(logits))


__all__ = [
    "OPTIMIZED_SSPA_GATE",
    "CENTERED_SSPA_GATE",
    "LEGACY_SSPA_GATE",
    "DEFAULT_SSPA_GATE",
    "SSPA_GATE_MODES",
    "strip_rms",
    "SSPA_BasicConv",
    "SSPA_ZPool",
    "SSPA_ChannelAttention",
    "SSPA_SpatialAttention",
    "SSPA_ZPoolChannelAttention",
    "SSPA_DirectionalStripPooling",
    "SSPA_StylePooling",
    "OptimizedSSPANet",
    "SSPANet_Block",
]
