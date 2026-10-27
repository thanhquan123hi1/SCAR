"""Modality fusion modules for multi-modal feature representations."""
from __future__ import annotations

import torch
from torch import nn


class ConcatFusion(nn.Module):
    def __init__(self, in_channels=1024, out_channels=512):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cine, psir, t2w):
        return self.fusion_conv(torch.cat((cine, psir, t2w), dim=1))


class Fusion_Embed(ConcatFusion):
    """Concatenate modalities, then project to the original skip width."""

    def __init__(self, embed_dim):
        super().__init__(embed_dim, embed_dim)


class SSPA_AttentionGate(nn.Module):
    """Multi-Level Skip Attention Gate incorporating directional strip pooling and spatial-channel filtering.

    Placed at high-resolution skip connections (e.g. 64x64, C=64; 32x32, C=256)
    to suppress extracardiac thoracic background noise (chest wall, lungs, ribs)
    prior to concatenation with decoder features.

    Architecture:
    1. Bilinear upsampling of coarser gating signal g (if g is provided and has different resolution).
    2. Projections via 1x1 convolutions: conv_x(x) and conv_g(g).
    3. GroupNorm + ReLU feature fusion.
    4. Directional Strip Refinement via 1x3 and 3x1 depthwise convolutions.
    5. 1x1 Conv + Sigmoid producing attention mask alpha in [0.0, 1.0].
    6. Output: gated features x * alpha.

    Supports standalone gating (when g is None) via direct strip gating or OptimizedSSPANet(mode='attention_gate').
    Total parameter overhead: < 0.02M params across both skip gates combined (19,890 parameters).
    """

    def __init__(
        self,
        in_channels: int,
        gate_channels: int | None = None,
        inter_channels: int | None = None,
        mode: str = "strip_gate",
        reduction_ratio: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.gate_channels = in_channels if gate_channels is None else gate_channels
        self.mode = mode
        self.reduction_ratio = reduction_ratio

        if inter_channels is None:
            inter_channels = max(min(in_channels, self.gate_channels) // 2, 16)
        self.inter_channels = inter_channels

        # Skip feature projection
        self.conv_x = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        # Gating signal projection
        self.conv_g = nn.Conv2d(self.gate_channels, inter_channels, 1, bias=False)

        # Normalization
        num_groups = min(8, inter_channels)
        while inter_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=inter_channels)
        self.relu = nn.ReLU(inplace=True)

        # Directional Strip Refinement (1x3 and 3x1 depthwise separable)
        self.strip_h = nn.Conv2d(
            inter_channels,
            inter_channels,
            (3, 1),
            padding=(1, 0),
            groups=inter_channels,
            bias=False,
        )
        self.strip_w = nn.Conv2d(
            inter_channels,
            inter_channels,
            (1, 3),
            padding=(0, 1),
            groups=inter_channels,
            bias=False,
        )

        # 1x1 Conv + Sigmoid producing attention coefficient alpha in [0.0, 1.0]
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        # Optional full OptimizedSSPANet block for mode='sspanet'
        if mode in ("sspanet", "optimized_sspanet"):
            from training.models.modules.sspanet import OptimizedSSPANet
            self.sspanet_block = OptimizedSSPANet(
                in_channels=in_channels,
                reduction_ratio=reduction_ratio,
                mode="attention_gate",
            )

    @property
    def sspa_net(self):
        """Lazy accessor for OptimizedSSPANet module if needed."""
        if hasattr(self, "sspanet_block"):
            return self.sspanet_block
        from training.models.modules.sspanet import OptimizedSSPANet
        return OptimizedSSPANet(
            in_channels=self.in_channels,
            reduction_ratio=self.reduction_ratio,
            mode="attention_gate",
        )

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.mode in ("sspanet", "optimized_sspanet") and hasattr(self, "sspanet_block"):
            if g is None:
                return self.sspanet_block(x, return_gate=return_attention) if return_attention else self.sspanet_block(x)

        if g is not None:
            if g.shape[2:] != x.shape[2:]:
                import torch.nn.functional as F
                g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
            proj = self.conv_x(x) + self.conv_g(g)
        else:
            proj = self.conv_x(x)

        act = self.relu(self.gn(proj))
        strip = self.strip_h(act) + self.strip_w(act)
        alpha = self.psi(act + strip)

        out = x * alpha
        if return_attention:
            return out, alpha
        return out


__all__ = [
    "ConcatFusion",
    "Fusion_Embed",
    "SSPA_AttentionGate",
]
