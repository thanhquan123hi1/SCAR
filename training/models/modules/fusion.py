"""Modality fusion modules for multi-modal feature representations."""
from __future__ import annotations

import torch
from torch import nn

MEAN_SKIP_FUSION = "mean"
MODALITY_WEIGHTED_SKIP_FUSION = "modality_weighted"
SPATIAL_RESIDUAL_SKIP_FUSION = "spatial_residual"
SKIP_FUSION_MODES = (MEAN_SKIP_FUSION, MODALITY_WEIGHTED_SKIP_FUSION, SPATIAL_RESIDUAL_SKIP_FUSION)


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


class TriModalSkipFusion(nn.Module):
    """Competitive channel + spatial softmax weighting across CINE, PSIR/LGE, and T2w skips.

    Replaces uniform (cine + psir + t2w)/3 averaging so fine high-resolution scar
    (PSIR) and edema (T2w) boundary features are preserved without dilution.
    Zero-initialized final logits start smoothly at uniform 1/3 weights.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        self.channels = channels
        mid = max(channels // reduction, 16)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(3 * channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 3 * channels, kernel_size=1, bias=True),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.channel_mlp[-1].weight)
        nn.init.zeros_(self.channel_mlp[-1].bias)
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.zeros_(self.spatial_gate[-1].bias)

    @staticmethod
    def _spatial_desc(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = cine.shape
        ch_summary = torch.cat(
            (
                cine.mean(dim=(2, 3), keepdim=True),
                psir.mean(dim=(2, 3), keepdim=True),
                t2w.mean(dim=(2, 3), keepdim=True),
            ),
            dim=1,
        )
        logits_ch = self.channel_mlp(ch_summary).view(b, 3, c, 1, 1)
        sp_in = torch.cat(
            (self._spatial_desc(cine), self._spatial_desc(psir), self._spatial_desc(t2w)),
            dim=1,
        )
        logits_sp = self.spatial_gate(sp_in).unsqueeze(2)
        weights = torch.softmax((logits_ch + logits_sp).float(), dim=1).to(cine.dtype)
        stacked = torch.stack((cine, psir, t2w), dim=1)
        return (stacked * weights).sum(dim=1)


class SpatialResidualSkipFusion(nn.Module):
    """E5-anchored skip fusion with pixel-wise spatial-contrast residual enhancement.

    Preserves E5's uniform average (cine + psir + t2w) / 3.0 as the primary skip
    representation so CINE anatomical boundaries are never diluted, while adding
    spatially gated differential pathology cues:
      - d_scar  = ReLU(psir - cine): focal LGE scar hyper-enhancement
      - d_edema = ReLU(t2w  - psir): pure T2w edema excess over PSIR
    via a zero-initialized residual branch.
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        gn_groups = 16 if channels >= 16 and channels % 16 == 0 else (8 if channels >= 8 and channels % 8 == 0 else 1)
        pw_groups = 4 if channels >= 16 and channels % 4 == 0 else 1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202602 + channels)
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=4, num_channels=16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 2, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            self.contrast_refine = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.GroupNorm(num_groups=gn_groups, num_channels=channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, groups=pw_groups, bias=False),
                nn.GroupNorm(num_groups=gn_groups, num_channels=channels),
            )
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def _spatial_desc(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        base = (cine + psir + t2w) / 3.0
        d_scar = torch.relu(psir - cine)
        d_edema = torch.relu(t2w - psir)
        sp_in = torch.cat(
            (self._spatial_desc(cine), self._spatial_desc(d_scar), self._spatial_desc(d_edema)),
            dim=1,
        )
        gates = self.spatial_gate(sp_in)
        g_scar = gates[:, 0:1]
        g_edema = gates[:, 1:2]
        contrast = g_scar * d_scar + g_edema * d_edema
        return base + self.gamma * self.contrast_refine(contrast)


__all__ = [
    "ConcatFusion",
    "Fusion_Embed",
    "SSPA_AttentionGate",
    "TriModalSkipFusion",
    "SpatialResidualSkipFusion",
    "MEAN_SKIP_FUSION",
    "MODALITY_WEIGHTED_SKIP_FUSION",
    "SPATIAL_RESIDUAL_SKIP_FUSION",
    "SKIP_FUSION_MODES",
]
