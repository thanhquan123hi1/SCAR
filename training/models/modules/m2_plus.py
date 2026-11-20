"""M2-Plus: Lightweight Spatial-Aware Decoupled Cross-Attention for SCAR.

Implements lightweight decoupled cross-attention for multi-sequence CMR myocardial
pathology segmentation (CINE, PSIR/LGE, and T2w).
Features:
- 2D Sinusoidal Positional Encoding (0 parameters) supplies spatial grid coordinates.
- Bottleneck latent dimension reduction (1024 -> 256) cutting cross-attention parameters
  from 9.97M to ~1.18M (-88.1% reduction).
- Decoupled Multi-Head Cross-Attention branches:
  * Branch 1: CINE Query attends to PSIR/LGE (Scar specialist)
  * Branch 2: CINE Query attends to T2w (Edema specialist)
- Historical full-width fusion is retained separately for old checkpoints.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

LEGACY_FUSION = "legacy_v1"
SPATIAL_FUSION = "spatial_v2"
FUSION_VERSIONS = (LEGACY_FUSION, SPATIAL_FUSION)

STANDARD_BOTTLENECK = "standard"
LOCAL_CONTRAST_BOTTLENECK = "local_contrast"
RESIDUAL_CONTRAST_BOTTLENECK = "residual_contrast"
RESIDUAL_CONTRAST_GATED_BOTTLENECK = "residual_contrast_gated"
RESIDUAL_CONTRAST_PROTO_BOTTLENECK = "residual_contrast_proto"
RESIDUAL_CONTRAST_COS_ATTN_BOTTLENECK = "residual_contrast_cos_attn"
MULTISCALE_CONTRAST_BOTTLENECK = "multiscale_contrast"
MULTISCALE_CONTRAST_SA_BOTTLENECK = "multiscale_contrast_sa"
MULTISCALE_CONTRAST_GATED_SA_BOTTLENECK = "multiscale_contrast_gated_sa"
BOTTLENECK_FUSION_MODES = (
    STANDARD_BOTTLENECK,
    LOCAL_CONTRAST_BOTTLENECK,
    RESIDUAL_CONTRAST_BOTTLENECK,
    RESIDUAL_CONTRAST_GATED_BOTTLENECK,
    RESIDUAL_CONTRAST_PROTO_BOTTLENECK,
    RESIDUAL_CONTRAST_COS_ATTN_BOTTLENECK,
    MULTISCALE_CONTRAST_BOTTLENECK,
    MULTISCALE_CONTRAST_SA_BOTTLENECK,
    MULTISCALE_CONTRAST_GATED_SA_BOTTLENECK,
)


def _get_group_norm_groups(channels: int, max_groups: int = 32) -> int:
    """Find the largest divisor of channels up to max_groups for GroupNorm."""
    for g in (max_groups, 16, 8, 4, 2, 1):
        if channels >= g and channels % g == 0:
            return g
    return 1


def build_2d_sinusoidal_positional_encoding(
    h: int,
    w: int,
    dim: int = 256,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    embed_dim: Optional[int] = None,
    temperature: float = 10000.0,
    normalize: bool = False,
) -> torch.Tensor:
    """Generate 2D sinusoidal positional encoding for spatial grid (H, W).

    Partitions dimension into vertical (d_y) and horizontal (d_x) components.
    Frequencies follow omega_i = temperature^(-4i / dim) = 10000^(-4i / dim).
    Resulting sinusoidal features are bounded within [-1.0, 1.0].

    Args:
        h: Height of the spatial grid.
        w: Width of the spatial grid.
        dim: Embedding dimension (must be divisible by 4). Default 256.
        device: Target torch device.
        dtype: Target torch data type.
        embed_dim: Optional alias for dim (for compatibility).
        temperature: Base for geometric progression of frequencies (default 10000.0).
        normalize: If True, uses continuous coordinates in [-1.0, 1.0].

    Returns:
        Positional encoding tensor of shape (1, H * W, dim).
    """
    if embed_dim is not None:
        dim = embed_dim
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D sin-cos PE, got {dim}")

    dim_y = dim // 2
    dim_x = dim - dim_y

    if normalize:
        grid_y = (
            torch.linspace(-1.0, 1.0, h, device=device, dtype=torch.float32)
            if h > 1
            else torch.zeros(1, device=device, dtype=torch.float32)
        )
        grid_x = (
            torch.linspace(-1.0, 1.0, w, device=device, dtype=torch.float32)
            if w > 1
            else torch.zeros(1, device=device, dtype=torch.float32)
        )
    else:
        grid_y = torch.arange(h, device=device, dtype=torch.float32)
        grid_x = torch.arange(w, device=device, dtype=torch.float32)

    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid_y = grid_y.flatten()
    grid_x = grid_x.flatten()

    omega_y = 1.0 / (temperature ** (torch.arange(0, dim_y, 2, device=device, dtype=torch.float32) / dim_y))
    omega_x = 1.0 / (temperature ** (torch.arange(0, dim_x, 2, device=device, dtype=torch.float32) / dim_x))

    out_y = torch.einsum("m,d->md", grid_y, omega_y)
    out_x = torch.einsum("m,d->md", grid_x, omega_x)

    pe_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    pe_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)

    pe_2d = torch.cat([pe_y, pe_x], dim=1)
    return pe_2d.unsqueeze(0).to(device=device, dtype=dtype)


# Alias for compatibility with test suite inspection
build_2d_sincos_position_embedding = build_2d_sinusoidal_positional_encoding


class SpatialAwareCrossAttention(nn.Module):
    """Lightweight Spatial-Aware Decoupled Cross-Attention for multi-sequence CMR.

    - Input channel projection: in_channels -> latent_dim (1x1 Conv + GroupNorm + ReLU)
    - Decoupled Cross-Attention:
        * Branch 1: CINE Query attends to PSIR/LGE Keys & Values (Scar Specialist)
        * Branch 2: CINE Query attends to T2w Keys & Values (Edema Specialist)
    - 2D Sinusoidal Positional Encoding injected into Query and Key tokens
    - Concat [CINE, Attn_Scar, Attn_Edema] -> 3 * latent_dim channels
    - Output projection: 3 * latent_dim -> out_channels (1x1 Conv + GroupNorm + ReLU)

    Budget: ~1.18M parameters (< 1.5M budget, -88.1% cut from 9.97M).
    """

    def __init__(
        self,
        in_channels: int = 1024,
        latent_dim: int = 256,
        out_channels: int = 512,
        num_heads: int = 8,
        use_pos_enc: bool = True,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        if in_channels < 1 or latent_dim < 1 or out_channels < 1 or num_heads < 1:
            raise ValueError("Attention channels and num_heads must be positive")
        if latent_dim % num_heads != 0:
            raise ValueError(f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})")
        if latent_dim % 4 != 0:
            raise ValueError(f"latent_dim ({latent_dim}) must be divisible by 4 for 2D sinusoidal PE")

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_pos_enc = use_pos_enc
        self.use_cross_attention = bool(use_cross_attention)

        # 1. Input projection: in_channels -> latent_dim
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_get_group_norm_groups(latent_dim, 32), num_channels=latent_dim),
            nn.ReLU(inplace=True),
        )

        # 2. Decoupled Multi-Head Cross-Attention modules (optional for ablation)
        if self.use_cross_attention:
            self.mha_scar = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
            self.mha_edema = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        else:
            self.mha_scar = None
            self.mha_edema = None

        # 3. Output projection: 3 * latent_dim -> out_channels
        self.proj_out = nn.Sequential(
            nn.Conv2d(3 * latent_dim, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_get_group_norm_groups(out_channels, 32), num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    @property
    def proj(self) -> nn.Module:
        """Backward compatibility alias for proj_out."""
        return self.proj_out

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        """Forward pass with spatial-aware decoupled cross-attention.

        Args:
            cine: CINE / bSSFP feature tensor (B, C_in, H, W).
            psir: PSIR / LGE feature tensor (B, C_in, H, W).
            t2w: T2w feature tensor (B, C_in, H, W).

        Returns:
            Fused feature tensor (B, out_channels, H, W).
        """
        b, _, h, w = cine.shape
        q_feat = self.proj_in(cine)
        k_scar = self.proj_in(psir)
        k_edema = self.proj_in(t2w)

        if not self.use_cross_attention:
            return self.proj_out(torch.cat([q_feat, k_scar, k_edema], dim=1))

        if self.use_pos_enc:
            pos_emb = build_2d_sinusoidal_positional_encoding(
                h, w, dim=self.latent_dim, device=cine.device, dtype=q_feat.dtype
            )
            q_tokens = q_feat.flatten(2).transpose(1, 2) + pos_emb
            k_scar_tokens = k_scar.flatten(2).transpose(1, 2) + pos_emb
            k_edema_tokens = k_edema.flatten(2).transpose(1, 2) + pos_emb
        else:
            q_tokens = q_feat.flatten(2).transpose(1, 2)
            k_scar_tokens = k_scar.flatten(2).transpose(1, 2)
            k_edema_tokens = k_edema.flatten(2).transpose(1, 2)

        v_scar_tokens = k_scar.flatten(2).transpose(1, 2)
        v_edema_tokens = k_edema.flatten(2).transpose(1, 2)

        attn_scar, _ = self.mha_scar(q_tokens, k_scar_tokens, v_scar_tokens, need_weights=False)
        attn_edema, _ = self.mha_edema(q_tokens, k_edema_tokens, v_edema_tokens, need_weights=False)

        attn_scar = attn_scar.transpose(1, 2).reshape(b, self.latent_dim, h, w)
        attn_edema = attn_edema.transpose(1, 2).reshape(b, self.latent_dim, h, w)

        fused = torch.cat([q_feat, attn_scar, attn_edema], dim=1)
        return self.proj_out(fused)


class LegacyM2PlusFusion(nn.Module):
    """Original full-width attention and BatchNorm, including historical state keys."""

    def __init__(self, in_channels=1024, out_channels=512, num_heads=8):
        super().__init__()
        if num_heads < 1 or in_channels % num_heads:
            raise ValueError("Channels must be divisible by positive num_heads")
        self.mha_scar = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.mha_edema = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.proj = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cine, psir, t2w):
        b, c, h, w = cine.shape
        query, scar, edema = [x.flatten(2).transpose(1, 2) for x in (cine, psir, t2w)]
        scar, _ = self.mha_scar(query, scar, scar, need_weights=False)
        edema, _ = self.mha_edema(query, edema, edema, need_weights=False)
        scar = scar.transpose(1, 2).reshape(b, c, h, w)
        edema = edema.transpose(1, 2).reshape(b, c, h, w)
        return self.proj(torch.cat((cine, scar, edema), dim=1))


class M2Plus_Fusion(SpatialAwareCrossAttention):
    """Spatial fusion accepting the historical constructor signature.

    Constructor compatibility does not convert full-width checkpoint weights;
    those require LegacyM2PlusFusion.

    Supports legacy positional signature:
        M2Plus_Fusion(in_channels, out_channels, num_heads)
    as well as modern keyword signature:
        M2Plus_Fusion(in_channels=1024, latent_dim=256, out_channels=512, num_heads=8, use_pos_enc=True)
    """

    def __init__(
        self,
        *args,
        in_channels: int = 1024,
        latent_dim: int = 256,
        out_channels: int = 512,
        num_heads: int = 8,
        use_pos_enc: bool = True,
        **kwargs,
    ):
        if len(args) == 1:
            in_channels = args[0]
        elif len(args) == 2:
            in_channels = args[0]
            out_channels = args[1]
        elif len(args) == 3:
            in_channels = args[0]
            out_channels = args[1]
            num_heads = args[2]
        elif len(args) >= 4:
            in_channels = args[0]
            latent_dim = args[1]
            out_channels = args[2]
            num_heads = args[3]

        if "latent_dim" in kwargs:
            latent_dim = kwargs.pop("latent_dim")
        if "out_channels" in kwargs:
            out_channels = kwargs.pop("out_channels")
        if "num_heads" in kwargs:
            num_heads = kwargs.pop("num_heads")
        if "use_pos_enc" in kwargs:
            use_pos_enc = kwargs.pop("use_pos_enc")

        super().__init__(
            in_channels=in_channels,
            latent_dim=latent_dim,
            out_channels=out_channels,
            num_heads=num_heads,
            use_pos_enc=use_pos_enc,
            **kwargs,
        )


class LocalModalityContrastFusion(nn.Module):
    """Modality-decoupled, contrast-aware local bottleneck fusion for aligned CMR.

    Uses per-modality 1x1 projections (groups=3) so CINE, PSIR/LGE, and T2w keep
    distinct semantic channels, injects explicit pathology difference cues
    (PSIR - CINE for Scar, T2w - PSIR for Edema), fuses them pointwise, and
    applies a local multi-scale depthwise residual block with channel SE gating.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        latent_dim: int = 192,
        out_channels: int = 512,
    ):
        super().__init__()
        if in_channels < 1 or latent_dim < 1 or out_channels < 1:
            raise ValueError("Bottleneck fusion channel dimensions must be positive")
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        g_mod = _get_group_norm_groups(latent_dim, 16)
        self.proj_mod = nn.Sequential(
            nn.Conv2d(3 * in_channels, 3 * latent_dim, kernel_size=1, groups=3, bias=False),
            nn.GroupNorm(num_groups=3 * g_mod, num_channels=3 * latent_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_fuse = nn.Sequential(
            nn.Conv2d(5 * latent_dim, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_get_group_norm_groups(out_channels, 32), num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

        mid = max(out_channels // 2, 16)
        g_mid = _get_group_norm_groups(mid, 16)
        se_mid = max(mid // 8, 8)
        self.reduce = nn.Sequential(
            nn.Conv2d(out_channels, mid, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=g_mid, num_channels=mid),
            nn.ReLU(inplace=True),
        )
        self.dw_local = nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False)
        self.dw_dilated = nn.Conv2d(mid, mid, kernel_size=3, padding=2, dilation=2, groups=mid, bias=False)
        self.norm_dw = nn.GroupNorm(num_groups=g_mid, num_channels=mid)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, se_mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_mid, mid, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_get_group_norm_groups(out_channels, 32), num_channels=out_channels),
        )

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        f_cine, f_psir, f_t2w = self.proj_mod(torch.cat((cine, psir, t2w), dim=1)).chunk(3, dim=1)
        d_scar = f_psir - f_cine
        d_edema = f_t2w - f_psir
        x = self.proj_fuse(torch.cat((f_cine, f_psir, f_t2w, d_scar, d_edema), dim=1))
        r = self.reduce(x)
        m = F.relu(self.norm_dw(self.dw_local(r) + self.dw_dilated(r)), inplace=True)
        m = m * self.se(m)
        return F.relu(x + self.expand(m), inplace=True)


class ResidualContrastBottleneck(nn.Module):
    """E5-anchored bottleneck fusion with zero-initialized differential contrast branch.

    Preserves E5's exact shared-basis projection (proj_in + proj_out) as the primary
    pathway and injects explicit rectified inter-modality differences in the aligned
    latent space:
      - d_scar  = ReLU(f_psir - f_cine): LGE/PSIR hyper-enhancement over CINE anatomy
      - d_edema = ReLU(f_t2w  - f_psir): Pure T2w edema excess over PSIR (suppresses Scar FP in Edema)
      - d_aar   = ReLU(f_t2w  - f_cine): Area-at-risk excess over CINE anatomy
    processed via a 3x3 depthwise-separable spatial context block with zero-init scale.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        latent_dim: int = 256,
        out_channels: int = 512,
    ):
        super().__init__()
        if in_channels < 1 or latent_dim < 1 or out_channels < 1:
            raise ValueError("Bottleneck fusion channel dimensions must be positive")
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_get_group_norm_groups(latent_dim, 32), num_channels=latent_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(latent_dim * 3, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_get_group_norm_groups(out_channels, 32), num_channels=out_channels),
            nn.ReLU(inplace=False),
        )

        mid = max(latent_dim // 2, 32)
        g_mid = _get_group_norm_groups(mid, 16)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202601 + in_channels + latent_dim + out_channels)
            self.contrast_branch = nn.Sequential(
                nn.Conv2d(latent_dim * 3, mid, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=g_mid, num_channels=mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False),
                nn.GroupNorm(num_groups=g_mid, num_channels=mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=_get_group_norm_groups(out_channels, 32), num_channels=out_channels),
            )
            self.contrast_gamma = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        q_feat = self.proj_in(cine)
        k_scar = self.proj_in(psir)
        k_edema = self.proj_in(t2w)
        base_out = self.proj_out(torch.cat((q_feat, k_scar, k_edema), dim=1))

        d_scar = F.relu(k_scar - q_feat)
        d_edema = F.relu(k_edema - k_scar)
        d_aar = F.relu(k_edema - q_feat)
        delta = self.contrast_branch(torch.cat((d_scar, d_edema, d_aar), dim=1))
        return F.relu(base_out + self.contrast_gamma * delta, inplace=True)


class MultiScaleSymmetricContrastBottleneck(nn.Module):
    """E5-anchored bottleneck with 4-way symmetric contrast, multi-scale 3x3 pyramid, and global slice gate.

    Upgrades ResidualContrastBottleneck (B1R) with:
      1. 4-way symmetric rectified differences in shared GroupNorm-aligned space:
         - d_scar_cine  = ReLU(f_psir - f_cine): LGE/PSIR hyper-enhancement over CINE
         - d_scar_t2    = ReLU(f_psir - f_t2w):  Pure Scar excess over T2w (suppresses Edema-as-Scar FP)
         - d_edema_pure = ReLU(f_t2w  - f_psir): Pure Edema excess over PSIR (suppresses Scar-as-Edema FP)
         - d_aar        = ReLU(f_t2w  - f_cine): Total Area-at-Risk excess over CINE
      2. Wide 256-ch multi-scale spatial pyramid (Full 3x3 Conv + Dilated 3x3 Conv + residual shortcut).
      3. Global Peak-to-Mean Slice Gate (compares slice peak against slice mean to suppress negative-slice FP).
    """

    def __init__(
        self,
        in_channels: int = 1024,
        latent_dim: int = 256,
        out_channels: int = 512,
    ):
        super().__init__()
        if in_channels < 1 or latent_dim < 1 or out_channels < 1:
            raise ValueError("Bottleneck fusion channel dimensions must be positive")
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_get_group_norm_groups(latent_dim, 32), num_channels=latent_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(latent_dim * 3, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_get_group_norm_groups(out_channels, 32), num_channels=out_channels),
            nn.ReLU(inplace=False),
        )

        mid = max(latent_dim, 64)
        g_mid = _get_group_norm_groups(mid, 32)
        g_out = _get_group_norm_groups(out_channels, 32)
        gate_hidden = max(mid // 4, 16)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202611 + in_channels + latent_dim + out_channels)
            self.compress = nn.Sequential(
                nn.Conv2d(latent_dim * 4, mid, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=g_mid, num_channels=mid),
                nn.ReLU(inplace=True),
            )
            self.ms_conv3x3 = nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False)
            self.ms_dilated = nn.Conv2d(mid, mid, kernel_size=3, padding=2, dilation=2, groups=mid, bias=False)
            self.ms_norm = nn.GroupNorm(num_groups=g_mid, num_channels=mid)
            self.slice_gate = nn.Sequential(
                nn.Conv2d(mid * 2, gate_hidden, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_hidden, mid, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            self.expand = nn.Sequential(
                nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=g_out, num_channels=out_channels),
            )
            self.contrast_gamma = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        q_feat = self.proj_in(cine)
        k_scar = self.proj_in(psir)
        k_edema = self.proj_in(t2w)
        base_out = self.proj_out(torch.cat((q_feat, k_scar, k_edema), dim=1))

        d_scar_cine = F.relu(k_scar - q_feat)
        d_scar_t2 = F.relu(k_scar - k_edema)
        d_edema_pure = F.relu(k_edema - k_scar)
        d_aar = F.relu(k_edema - q_feat)

        c = self.compress(torch.cat((d_scar_cine, d_scar_t2, d_edema_pure, d_aar), dim=1))
        m = F.relu(self.ms_norm(self.ms_conv3x3(c) + self.ms_dilated(c) + c), inplace=True)
        m_avg = m.mean(dim=(2, 3), keepdim=True)
        m_max = m.amax(dim=(2, 3), keepdim=True)
        g = self.slice_gate(torch.cat((m_avg, m_max - m_avg), dim=1))
        delta = self.expand(m * g)
        return F.relu(base_out + self.contrast_gamma * delta, inplace=True)


class GatedSymmetricContrastBottleneck(nn.Module):
    """Lightweight B1R bottleneck upgraded with 4-way symmetric contrast and Peak-to-Mean slice gating."""

    def __init__(
        self,
        in_channels: int = 1024,
        latent_dim: int = 256,
        out_channels: int = 512,
    ):
        super().__init__()
        if in_channels < 1 or latent_dim < 1 or out_channels < 1:
            raise ValueError("Bottleneck fusion channel dimensions must be positive")
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_get_group_norm_groups(latent_dim, 32), num_channels=latent_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(latent_dim * 3, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_get_group_norm_groups(out_channels, 32), num_channels=out_channels),
            nn.ReLU(inplace=False),
        )

        mid = max(latent_dim // 2, 32)
        g_mid = _get_group_norm_groups(mid, 16)
        gate_hidden = max(mid // 4, 16)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202651 + in_channels + latent_dim + out_channels)
            self.compress = nn.Sequential(
                nn.Conv2d(latent_dim * 4, mid, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=g_mid, num_channels=mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False),
                nn.GroupNorm(num_groups=g_mid, num_channels=mid),
                nn.ReLU(inplace=True),
            )
            self.slice_gate = nn.Sequential(
                nn.Conv2d(mid * 2, gate_hidden, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_hidden, mid, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            self.expand = nn.Sequential(
                nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=_get_group_norm_groups(out_channels, 32), num_channels=out_channels),
            )
            self.contrast_gamma = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        q_feat = self.proj_in(cine)
        k_scar = self.proj_in(psir)
        k_edema = self.proj_in(t2w)
        base_out = self.proj_out(torch.cat((q_feat, k_scar, k_edema), dim=1))

        d_scar_cine = F.relu(k_scar - q_feat)
        d_scar_t2 = F.relu(k_scar - k_edema)
        d_edema_pure = F.relu(k_edema - k_scar)
        d_aar = F.relu(k_edema - q_feat)

        m = self.compress(torch.cat((d_scar_cine, d_scar_t2, d_edema_pure, d_aar), dim=1))
        m_avg = m.mean(dim=(2, 3), keepdim=True)
        m_max = m.amax(dim=(2, 3), keepdim=True)
        g = self.slice_gate(torch.cat((m_avg, m_max - m_avg), dim=1))
        delta = self.expand(m * g)
        return F.relu(base_out + self.contrast_gamma * delta, inplace=True)


class MultiScaleContrastSelfAttnBottleneck(MultiScaleSymmetricContrastBottleneck):
    """B2 bottleneck followed by zero-initialized 2D positional spatial self-attention.

    Unlike cross-modality MHA (which broke pixel-to-pixel alignment between CINE,
    PSIR, and T2w), self-attention operates *after* pixel-aligned multi-modal fusion
    so every 8x8 sector compares its aligned pathology signature against all other
    sectors in the slice — suppressing disconnected opposite-wall T2w shading FPs
    and enforcing global closed myocardial ring consistency.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        latent_dim: int = 256,
        out_channels: int = 512,
        num_heads: int = 8,
    ):
        super().__init__(in_channels=in_channels, latent_dim=latent_dim, out_channels=out_channels)
        sa_dim = max(latent_dim, 64)
        # Ensure divisibility by 4 (for 2D sin-cos PE) and num_heads
        while sa_dim % num_heads != 0 or sa_dim % 4 != 0:
            num_heads = max(num_heads // 2, 1)
        self.sa_dim = sa_dim
        self.sa_heads = num_heads
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202711 + in_channels + latent_dim + out_channels)
            self.sa_in = nn.Sequential(
                nn.Conv2d(out_channels, sa_dim, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=_get_group_norm_groups(sa_dim, 32), num_channels=sa_dim),
            )
            self.sa_norm = nn.LayerNorm(sa_dim)
            self.sa_mha = nn.MultiheadAttention(sa_dim, num_heads, batch_first=True)
            self.sa_out = nn.Sequential(
                nn.Conv2d(sa_dim, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=_get_group_norm_groups(out_channels, 32), num_channels=out_channels),
            )
            self.sa_gamma = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        fused = super().forward(cine, psir, t2w)
        b, _, h, w = fused.shape
        x = self.sa_in(fused)
        pos = build_2d_sinusoidal_positional_encoding(h, w, dim=self.sa_dim, device=x.device, dtype=x.dtype)
        tokens = x.flatten(2).transpose(1, 2)
        qkv = self.sa_norm(tokens) + pos
        attn_out, _ = self.sa_mha(qkv, qkv, tokens, need_weights=False)
        attn_map = attn_out.transpose(1, 2).reshape(b, self.sa_dim, h, w)
        delta = self.sa_out(attn_map)
        return F.relu(fused + self.sa_gamma * delta, inplace=True)


class MultiScaleContrastGatedSelfAttnBottleneck(MultiScaleContrastSelfAttnBottleneck):
    """B2SA bottleneck where 2D spatial self-attention is gated by B2's Peak-to-Mean slice gate.

    Prevents self-attention from re-injecting ungated background/coil noise on
    low-peak or empty-scar slices (e.g., Case3003 z=2 and Case3026 z=4) while
    preserving full cross-wall spatial comparison on pathology-bearing slices.
    """

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        q_feat = self.proj_in(cine)
        k_scar = self.proj_in(psir)
        k_edema = self.proj_in(t2w)
        base_out = self.proj_out(torch.cat((q_feat, k_scar, k_edema), dim=1))

        d_scar_cine = F.relu(k_scar - q_feat)
        d_scar_t2 = F.relu(k_scar - k_edema)
        d_edema_pure = F.relu(k_edema - k_scar)
        d_aar = F.relu(k_edema - q_feat)

        c = self.compress(torch.cat((d_scar_cine, d_scar_t2, d_edema_pure, d_aar), dim=1))
        m = F.relu(self.ms_norm(self.ms_conv3x3(c) + self.ms_dilated(c) + c), inplace=True)
        m_avg = m.mean(dim=(2, 3), keepdim=True)
        m_max = m.amax(dim=(2, 3), keepdim=True)
        g = self.slice_gate(torch.cat((m_avg, m_max - m_avg), dim=1))
        delta_b2 = self.expand(m * g)
        fused = F.relu(base_out + self.contrast_gamma * delta_b2, inplace=True)

        b, _, h, w = fused.shape
        x = self.sa_in(fused)
        pos = build_2d_sinusoidal_positional_encoding(h, w, dim=self.sa_dim, device=x.device, dtype=x.dtype)
        tokens = x.flatten(2).transpose(1, 2)
        qkv = self.sa_norm(tokens) + pos
        attn_out, _ = self.sa_mha(qkv, qkv, tokens, need_weights=False)
        attn_map = attn_out.transpose(1, 2).reshape(b, self.sa_dim, h, w)
        g_slice = g.mean(dim=1, keepdim=True)
        delta_sa = self.sa_out(attn_map) * g_slice
        return F.relu(fused + self.sa_gamma * delta_sa, inplace=True)


class ResidualProtoContrastBottleneck(ResidualContrastBottleneck):
    """E5-B1R bottleneck enhanced with CINE-guided myocardial prototype contrast calibration.

    Motivated by RPT (MICCAI 2023) and GLFNet (2024):
    1. Extracts a soft myocardial wall mask w_myo from CINE bottleneck features q_feat.
    2. Pools CINE-weighted regional myocardial prototypes proto_myo for [d_scar, d_edema, d_aar]
       to subtract slice-wide low-frequency T2w/PSIR coil-shading offsets across the myocardial ring
       without dense 64x64 background token mixing.
    3. Refines prototype-centered focal contrast via a zero-initialized residual branch (proto_gamma=0),
       preserving exact step-0 equivalence with ResidualContrastBottleneck (B1R).
    """

    def __init__(
        self,
        in_channels: int = 1024,
        latent_dim: int = 256,
        out_channels: int = 512,
    ):
        super().__init__(in_channels=in_channels, latent_dim=latent_dim, out_channels=out_channels)
        mid = max(latent_dim // 2, 32)
        g_mid = _get_group_norm_groups(mid, 16)
        gate_mid = max(latent_dim // 4, 16)
        g_gate = _get_group_norm_groups(gate_mid, 8)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202731 + in_channels + latent_dim + out_channels)
            self.myo_gate = nn.Sequential(
                nn.Conv2d(latent_dim, gate_mid, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=g_gate, num_channels=gate_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_mid, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            self.proto_scale = nn.Parameter(torch.full((1, latent_dim * 3, 1, 1), 0.5))
            self.proto_branch = nn.Sequential(
                nn.Conv2d(latent_dim * 3, mid, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=g_mid, num_channels=mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False),
                nn.GroupNorm(num_groups=g_mid, num_channels=mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=_get_group_norm_groups(out_channels, 32), num_channels=out_channels),
            )
            self.proto_gamma = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        q_feat = self.proj_in(cine)
        k_scar = self.proj_in(psir)
        k_edema = self.proj_in(t2w)
        base_out = self.proj_out(torch.cat((q_feat, k_scar, k_edema), dim=1))

        d_scar = F.relu(k_scar - q_feat)
        d_edema = F.relu(k_edema - k_scar)
        d_aar = F.relu(k_edema - q_feat)
        diff_cat = torch.cat((d_scar, d_edema, d_aar), dim=1)
        delta_b1r = self.contrast_branch(diff_cat)

        w_myo = self.myo_gate(q_feat)
        w_sum = w_myo.sum(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        proto_myo = (diff_cat * w_myo).sum(dim=(2, 3), keepdim=True) / w_sum
        scale = self.proto_scale.clamp(0.0, 1.5)
        focal_diff = F.relu(diff_cat - scale * proto_myo) * (0.5 + 0.5 * w_myo)
        delta_proto = self.proto_branch(focal_diff)

        return F.relu(
            base_out + self.contrast_gamma * delta_b1r + self.proto_gamma * delta_proto,
            inplace=True,
        )


class ResidualContrastCosAttnBottleneck(nn.Module):
    """E5/B1R-anchored bottleneck with I-MMSeg-inspired parameter-free cross-modal cosine attention.

    Preserves ResidualContrastBottleneck (B1R) exactly as the primary pathway and adds a
    lightweight, zero-initialized global cross-modal cosine-attention branch at the 8x8
    bottleneck (no heavy Q/K/V projection matrices to avoid overfitting with Karimi HD-DT):
      1. Slice-conditioned GAP spatial importance masks suppress background tokens per modality.
      2. L2-normalized pairwise cross-modal |cosine similarity| correlates tokens across the
         full myocardial ring (e.g., comparing opposite myocardial walls).
      3. Cross-modal aggregated features are fused and added via a zero-initialized scale.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        latent_dim: int = 256,
        out_channels: int = 512,
    ):
        super().__init__()
        if in_channels < 1 or latent_dim < 1 or out_channels < 1:
            raise ValueError("Bottleneck fusion channel dimensions must be positive")
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_get_group_norm_groups(latent_dim, 32), num_channels=latent_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(latent_dim * 3, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_get_group_norm_groups(out_channels, 32), num_channels=out_channels),
            nn.ReLU(inplace=False),
        )

        mid = max(latent_dim // 2, 32)
        g_mid = _get_group_norm_groups(mid, 16)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202601 + in_channels + latent_dim + out_channels)
            self.contrast_branch = nn.Sequential(
                nn.Conv2d(latent_dim * 3, mid, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=g_mid, num_channels=mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False),
                nn.GroupNorm(num_groups=g_mid, num_channels=mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=_get_group_norm_groups(out_channels, 32), num_channels=out_channels),
            )
            self.contrast_gamma = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        gate_mid = max(latent_dim // 4, 32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202641 + in_channels + latent_dim + out_channels)
            self.mask_ctx = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(latent_dim, gate_mid, kernel_size=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(gate_mid, latent_dim, kernel_size=1, bias=True),
                    nn.Tanh(),
                )
                for _ in range(3)
            ])
            self.mask_spatial = nn.ModuleList([
                nn.Conv2d(latent_dim, 1, kernel_size=1, bias=True)
                for _ in range(3)
            ])
            # Softplus(1.0) + 0.5 ≈ 1.81 initial cosine-attention temperature
            self.cos_log_temp = nn.Parameter(torch.tensor(1.0))
            self.cos_fuse = nn.Sequential(
                nn.Conv2d(latent_dim * 3, mid, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=g_mid, num_channels=mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=_get_group_norm_groups(out_channels, 32), num_channels=out_channels),
            )
            self.cos_gamma = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    @staticmethod
    def _cos_attn_update(val_bcn: torch.Tensor, sim_bnn: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(sim_bnn * temp, dim=-1)  # (B, N, N)
        # (B, N, N) @ (B, N, C) -> (B, N, C) -> (B, C, N)
        return torch.bmm(attn, val_bcn.transpose(1, 2)).transpose(1, 2)

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        q_feat = self.proj_in(cine)
        k_scar = self.proj_in(psir)
        k_edema = self.proj_in(t2w)
        base_out = self.proj_out(torch.cat((q_feat, k_scar, k_edema), dim=1))

        d_scar = F.relu(k_scar - q_feat)
        d_edema = F.relu(k_edema - k_scar)
        d_aar = F.relu(k_edema - q_feat)
        delta_b1r = self.contrast_branch(torch.cat((d_scar, d_edema, d_aar), dim=1))

        # Slice-conditioned spatial masking + parameter-free cross-modal cosine attention (fp32 for bf16 safety)
        b, c, h, w = q_feat.shape
        mods = (q_feat, k_scar, k_edema)
        masked = []
        normed = []
        for idx, feat in enumerate(mods):
            ctx = self.mask_ctx[idx](feat)
            sp_mask = torch.sigmoid(self.mask_spatial[idx](feat * (1.0 + ctx)))
            u = (feat * sp_mask).float().flatten(2)  # (B, C, N)
            masked.append(u)
            normed.append(F.normalize(u, p=2, dim=1, eps=1e-6))

        u_c, u_p, u_t = masked
        n_c, n_p, n_t = normed
        sim_cp = torch.abs(torch.bmm(n_c.transpose(1, 2), n_p))  # (B, N, N)
        sim_ct = torch.abs(torch.bmm(n_c.transpose(1, 2), n_t))
        sim_pt = torch.abs(torch.bmm(n_p.transpose(1, 2), n_t))

        temp = F.softplus(self.cos_log_temp.float()) + 0.5
        c_upd = 0.5 * (self._cos_attn_update(u_p, sim_cp, temp) + self._cos_attn_update(u_t, sim_ct, temp))
        p_upd = 0.5 * (self._cos_attn_update(u_c, sim_cp.transpose(1, 2), temp) + self._cos_attn_update(u_t, sim_pt, temp))
        t_upd = 0.5 * (self._cos_attn_update(u_c, sim_ct.transpose(1, 2), temp) + self._cos_attn_update(u_p, sim_pt.transpose(1, 2), temp))

        cos_cat = torch.cat((c_upd, p_upd, t_upd), dim=1).reshape(b, 3 * c, h, w).to(base_out.dtype)
        delta_cos = self.cos_fuse(cos_cat)

        return F.relu(
            base_out + self.contrast_gamma * delta_b1r + self.cos_gamma * delta_cos,
            inplace=True,
        )


__all__ = [
    "LEGACY_FUSION", "SPATIAL_FUSION", "FUSION_VERSIONS", "LegacyM2PlusFusion",
    "STANDARD_BOTTLENECK", "LOCAL_CONTRAST_BOTTLENECK", "RESIDUAL_CONTRAST_BOTTLENECK",
    "RESIDUAL_CONTRAST_GATED_BOTTLENECK", "RESIDUAL_CONTRAST_PROTO_BOTTLENECK",
    "RESIDUAL_CONTRAST_COS_ATTN_BOTTLENECK",
    "MULTISCALE_CONTRAST_BOTTLENECK",
    "MULTISCALE_CONTRAST_SA_BOTTLENECK", "MULTISCALE_CONTRAST_GATED_SA_BOTTLENECK",
    "BOTTLENECK_FUSION_MODES",
    "SpatialAwareCrossAttention",
    "LocalModalityContrastFusion",
    "ResidualContrastBottleneck",
    "GatedSymmetricContrastBottleneck",
    "ResidualProtoContrastBottleneck",
    "ResidualContrastCosAttnBottleneck",
    "MultiScaleSymmetricContrastBottleneck",
    "MultiScaleContrastSelfAttnBottleneck",
    "MultiScaleContrastGatedSelfAttnBottleneck",
    "M2Plus_Fusion",
    "build_2d_sinusoidal_positional_encoding",
    "build_2d_sincos_position_embedding",
]
