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
BOTTLENECK_FUSION_MODES = (STANDARD_BOTTLENECK, LOCAL_CONTRAST_BOTTLENECK)


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


__all__ = [
    "LEGACY_FUSION", "SPATIAL_FUSION", "FUSION_VERSIONS", "LegacyM2PlusFusion",
    "STANDARD_BOTTLENECK", "LOCAL_CONTRAST_BOTTLENECK", "BOTTLENECK_FUSION_MODES",
    "SpatialAwareCrossAttention",
    "LocalModalityContrastFusion",
    "M2Plus_Fusion",
    "build_2d_sinusoidal_positional_encoding",
    "build_2d_sincos_position_embedding",
]
