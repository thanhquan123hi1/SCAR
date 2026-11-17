"""Modality fusion modules for multi-modal feature representations."""
from __future__ import annotations

import torch
from torch import nn

MEAN_SKIP_FUSION = "mean"
MODALITY_WEIGHTED_SKIP_FUSION = "modality_weighted"
SPATIAL_RESIDUAL_SKIP_FUSION = "spatial_residual"
NORM_CONTRAST_SKIP_FUSION = "norm_contrast"
DECOUPLED_ANAT_CONTRAST_SKIP_FUSION = "decoupled_anat_contrast"
NORM_CONTRAST_ANAT_MEAN_SKIP_FUSION = "norm_contrast_anat_mean"
NORM_CONTRAST_MS_SKIP_FUSION = "norm_contrast_ms"
NORM_CONTRAST_ANAT_MEAN_MS_SKIP_FUSION = "norm_contrast_anat_mean_ms"
SKIP_FUSION_MODES = (
    MEAN_SKIP_FUSION,
    MODALITY_WEIGHTED_SKIP_FUSION,
    SPATIAL_RESIDUAL_SKIP_FUSION,
    NORM_CONTRAST_SKIP_FUSION,
    DECOUPLED_ANAT_CONTRAST_SKIP_FUSION,
    NORM_CONTRAST_ANAT_MEAN_SKIP_FUSION,
    NORM_CONTRAST_MS_SKIP_FUSION,
    NORM_CONTRAST_ANAT_MEAN_MS_SKIP_FUSION,
)


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


class NormalizedContrastSkipFusion(nn.Module):
    """E5-anchored skip fusion with GroupNorm-aligned 4-way symmetric contrast and full 3x3 fusion.

    Aligns CINE, PSIR, and T2w skip features via a shared 1x1 Conv + GroupNorm + ReLU
    projection before computing 4-way rectified differences (d_scar_cine, d_scar_t2,
    d_edema_pure, d_aar), processes them through a full 3x3 spatial context block,
    and injects the result into E5's uniform skip mean via a zero-initialized scale.
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        mid = max(channels // 2, 32)
        gn_mid = 16 if mid >= 16 and mid % 16 == 0 else (8 if mid >= 8 and mid % 8 == 0 else 1)
        gn_out = 16 if channels >= 16 and channels % 16 == 0 else (8 if channels >= 8 and channels % 8 == 0 else 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202622 + channels)
            self.align_proj = nn.Sequential(
                nn.Conv2d(channels, mid, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=gn_mid, num_channels=mid),
                nn.ReLU(inplace=True),
            )
            self.contrast_fuse = nn.Sequential(
                nn.Conv2d(mid * 4, mid, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=gn_mid, num_channels=mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=gn_mid, num_channels=mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=gn_out, num_channels=channels),
            )
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        base = (cine + psir + t2w) / 3.0
        q_cine = self.align_proj(cine)
        k_psir = self.align_proj(psir)
        k_t2w = self.align_proj(t2w)

        d_scar_cine = torch.relu(k_psir - q_cine)
        d_scar_t2 = torch.relu(k_psir - k_t2w)
        d_edema_pure = torch.relu(k_t2w - k_psir)
        d_aar = torch.relu(k_t2w - q_cine)

        delta = self.contrast_fuse(
            torch.cat((d_scar_cine, d_scar_t2, d_edema_pure, d_aar), dim=1)
        )
        return base + self.gamma * delta


def _gn_groups(channels: int, max_groups: int = 16) -> int:
    for g in (max_groups, 8, 4, 2, 1):
        if channels >= g and channels % g == 0:
            return g
    return 1


class _DecoupledContrastBranch(nn.Module):
    """Lightweight 1x1 -> 3x3 -> 1x1 skip contrast branch with optional Peak-to-Mean slice gate."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, use_slice_gate: bool = False):
        super().__init__()
        gn_mid = _gn_groups(mid_ch, 16)
        gn_out = _gn_groups(out_ch, 16)
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=gn_mid, num_channels=mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=gn_mid, num_channels=mid_ch),
            nn.ReLU(inplace=True),
        )
        if use_slice_gate:
            gate_hidden = max(mid_ch // 4, 16)
            self.slice_gate = nn.Sequential(
                nn.Conv2d(mid_ch * 2, gate_hidden, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_hidden, mid_ch, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.slice_gate = None
        self.expand = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=gn_out, num_channels=out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.reduce_conv(x)
        if self.slice_gate is not None:
            h_avg = h.mean(dim=(2, 3), keepdim=True)
            h_max = h.amax(dim=(2, 3), keepdim=True)
            h = h * self.slice_gate(torch.cat((h_avg, h_max - h_avg), dim=1))
        return self.expand(h)


class DecoupledAnatContrastSkipFusion(nn.Module):
    """E5-anchored skip fusion with modality-specific projections, CINE wall gating, and decoupled Scar/Edema branches.

    Fixes two weaknesses of NormalizedContrastSkipFusion (S3):
    1. Uses grouped 1x1 projections (groups=3) so CINE, PSIR, and T2w keep distinct semantics.
    2. Gates pathology contrast cues by a CINE-anchored myocardial wall mask (g_wall) and separates
       Scar (with PSIR-T2w harmonic co-activation + Peak-to-Mean slice gate) from Edema via independent
       zero-initialized gamma scales so Edema gains never erode Scar recall.
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        mid = max(channels // 2, 32)
        mid_b = max(mid // 2, 24)
        gn_mid = _gn_groups(mid, 16)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202642 + channels)
            self.align_proj = nn.Sequential(
                nn.Conv2d(channels * 3, mid * 3, kernel_size=1, groups=3, bias=False),
                nn.GroupNorm(num_groups=3 * gn_mid, num_channels=mid * 3),
                nn.ReLU(inplace=True),
            )
            self.wall_gate = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=4, num_channels=16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            self.scar_branch = _DecoupledContrastBranch(mid * 3, mid_b, channels, use_slice_gate=True)
            self.edema_branch = _DecoupledContrastBranch(mid * 2, mid_b, channels, use_slice_gate=False)
            self.gamma_scar = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.gamma_edema = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def _spatial_desc(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        base = (cine + psir + t2w) / 3.0
        q_cine, k_psir, k_t2w = self.align_proj(torch.cat((cine, psir, t2w), dim=1)).chunk(3, dim=1)
        g_wall = self.wall_gate(torch.cat((self._spatial_desc(q_cine), self._spatial_desc(base)), dim=1))

        d_scar_cine = torch.relu(k_psir - q_cine)
        d_scar_t2 = torch.relu(k_psir - k_t2w)
        # Harmonic co-activation: bounded by min(k_psir, k_t2w) and suppressed in dark-T2w blood pool.
        coact_scar = (2.0 * k_psir * k_t2w) / (k_psir + k_t2w + 1e-2)
        scar_in = torch.cat((d_scar_cine, d_scar_t2, coact_scar), dim=1) * g_wall

        d_edema_pure = torch.relu(k_t2w - k_psir)
        d_aar = torch.relu(k_t2w - q_cine)
        edema_in = torch.cat((d_edema_pure, d_aar), dim=1) * g_wall

        delta_scar = self.scar_branch(scar_in)
        delta_edema = self.edema_branch(edema_in)
        return base + self.gamma_scar * delta_scar + self.gamma_edema * delta_edema


def _build_align_proj(channels: int, mid: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, mid, kernel_size=1, bias=False),
        nn.GroupNorm(num_groups=_gn_groups(mid, 16), num_channels=mid),
        nn.ReLU(inplace=True),
    )


def _build_cine_wall_gate() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=4, num_channels=16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 1, kernel_size=1, bias=True),
        nn.Sigmoid(),
    )


def _compute_4way_contrasts(
    q_cine: torch.Tensor, k_psir: torch.Tensor, k_t2w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.relu(k_psir - q_cine),
        torch.relu(k_psir - k_t2w),
        torch.relu(k_t2w - k_psir),
        torch.relu(k_t2w - q_cine),
    )


class _BaseAnatMeanContrastSkipFusion(nn.Module):
    """Shared wall-gated myocardial-mean contrast extraction for S3AnatMean skip variants."""

    def _init_anat_mean_dims(self, channels: int) -> tuple[int, int, int]:
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        mid = max(channels // 2, 32)
        return mid, _gn_groups(mid, 16), _gn_groups(channels, 16)

    def _init_anat_mean_projections(self, channels: int, mid: int) -> None:
        self.align_proj = _build_align_proj(channels, mid)
        self.wall_gate = _build_cine_wall_gate()

    @staticmethod
    def _wall_mean(feat: torch.Tensor, weight_fp32: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        return ((feat.float() * weight_fp32).sum(dim=(2, 3), keepdim=True) / denom).to(feat.dtype)

    def _anat_mean_features(
        self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = (cine + psir + t2w) / 3.0
        q_cine = self.align_proj(cine)
        k_psir = self.align_proj(psir)
        k_t2w = self.align_proj(t2w)

        sp = torch.cat(
            (DecoupledAnatContrastSkipFusion._spatial_desc(q_cine), DecoupledAnatContrastSkipFusion._spatial_desc(base)),
            dim=1,
        )
        g_wall = self.wall_gate(sp)
        w_fp32 = g_wall.float()
        denom = w_fp32.sum(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        mu_psir = self._wall_mean(k_psir, w_fp32, denom)
        mu_t2w = self._wall_mean(k_t2w, w_fp32, denom)

        d_scar_cine, d_scar_t2, d_edema_pure, d_aar = _compute_4way_contrasts(q_cine, k_psir, k_t2w)
        d_psir_wall = torch.relu(k_psir - mu_psir) * g_wall
        d_t2w_wall = torch.relu(k_t2w - mu_t2w) * g_wall
        feat = torch.cat((d_scar_cine, d_scar_t2, d_edema_pure, d_aar, d_psir_wall, d_t2w_wall, g_wall), dim=1)
        return base, feat


class _MultiScaleHeadMixin:
    """Shared 1x1 compress -> (3x3 d=1 + DW 3x3 d=2) -> 1x1 expand head with zero-init gamma."""

    def _init_ms_head(self, in_channels: int, mid: int, out_channels: int, gn_mid: int, gn_out: int) -> None:
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=gn_mid, num_channels=mid),
            nn.ReLU(inplace=True),
        )
        self.conv_local = nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False)
        self.conv_dilated = nn.Conv2d(mid, mid, kernel_size=3, padding=2, dilation=2, groups=mid, bias=False)
        self.ms_norm = nn.GroupNorm(num_groups=gn_mid, num_channels=mid)
        self.ms_relu = nn.ReLU(inplace=True)
        self.expand = nn.Sequential(
            nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=gn_out, num_channels=out_channels),
        )
        self.gamma = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def _forward_ms_head(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.compress(feat)
        ms = self.ms_relu(self.ms_norm(self.conv_local(h) + self.conv_dilated(h)))
        return self.expand(ms)


class AnatMeanNormalizedContrastSkipFusion(_BaseAnatMeanContrastSkipFusion):
    """Shared-basis S3 skip fusion enriched with wall-gated myocardial-mean deviation.

    Preserves NormalizedContrastSkipFusion (S3)'s shared align_proj and unified
    1x1 -> full 3x3 -> 1x1 contrast block so CINE/PSIR/T2w keep matched channel
    semantics and Scar/Edema/NormalMyo interact jointly, while adding:
      - CINE-anchored soft myocardial wall gate g_wall (suppresses bright outer blood pool)
      - Wall-mean-centered PSIR & T2w hyper-intensity (no /std noise amplification,
        and stays positive inside Scar even when raw PSIR < CINE as in dark-core cases).
    """

    def __init__(self, channels: int):
        super().__init__()
        mid, gn_mid, gn_out = self._init_anat_mean_dims(channels)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202672 + channels)
            self._init_anat_mean_projections(channels, mid)
            self.contrast_fuse = nn.Sequential(
                nn.Conv2d(mid * 6 + 1, mid, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=gn_mid, num_channels=mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=gn_mid, num_channels=mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=gn_out, num_channels=channels),
            )
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        base, feat = self._anat_mean_features(cine, psir, t2w)
        return base + self.gamma * self.contrast_fuse(feat)


class MultiScaleNormalizedContrastSkipFusion(_MultiScaleHeadMixin, nn.Module):
    """Multi-scale S3 skip fusion (3x3 d=1 + dilated 3x3 d=2) in shared aligned space."""

    def __init__(self, channels: int):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        mid = max(channels // 2, 32)
        gn_mid = _gn_groups(mid, 16)
        gn_out = _gn_groups(channels, 16)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202682 + channels)
            self.align_proj = _build_align_proj(channels, mid)
            self._init_ms_head(mid * 4, mid, channels, gn_mid, gn_out)

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        base = (cine + psir + t2w) / 3.0
        q_cine = self.align_proj(cine)
        k_psir = self.align_proj(psir)
        k_t2w = self.align_proj(t2w)
        c4 = _compute_4way_contrasts(q_cine, k_psir, k_t2w)
        delta = self._forward_ms_head(torch.cat(c4, dim=1))
        return base + self.gamma * delta


class AnatMeanMSNormalizedContrastSkipFusion(_MultiScaleHeadMixin, _BaseAnatMeanContrastSkipFusion):
    """Wave 7B: wall-gated myocardial-mean S3 skip fusion with multi-scale (3x3 + DW 3x3 d=2) context."""

    def __init__(self, channels: int):
        super().__init__()
        mid, gn_mid, gn_out = self._init_anat_mean_dims(channels)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202692 + channels)
            self._init_anat_mean_projections(channels, mid)
            self._init_ms_head(mid * 6 + 1, mid, channels, gn_mid, gn_out)

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        base, feat = self._anat_mean_features(cine, psir, t2w)
        return base + self.gamma * self._forward_ms_head(feat)


__all__ = [
    "ConcatFusion",
    "Fusion_Embed",
    "SSPA_AttentionGate",
    "TriModalSkipFusion",
    "SpatialResidualSkipFusion",
    "NormalizedContrastSkipFusion",
    "DecoupledAnatContrastSkipFusion",
    "AnatMeanNormalizedContrastSkipFusion",
    "MultiScaleNormalizedContrastSkipFusion",
    "AnatMeanMSNormalizedContrastSkipFusion",
    "MEAN_SKIP_FUSION",
    "MODALITY_WEIGHTED_SKIP_FUSION",
    "SPATIAL_RESIDUAL_SKIP_FUSION",
    "NORM_CONTRAST_SKIP_FUSION",
    "DECOUPLED_ANAT_CONTRAST_SKIP_FUSION",
    "NORM_CONTRAST_ANAT_MEAN_SKIP_FUSION",
    "NORM_CONTRAST_MS_SKIP_FUSION",
    "NORM_CONTRAST_ANAT_MEAN_MS_SKIP_FUSION",
    "SKIP_FUSION_MODES",
]
