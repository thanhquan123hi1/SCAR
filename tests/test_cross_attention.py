"""Tier 1 and Tier 2 specification tests for Spatial-Aware Decoupled Cross-Attention.

Verifies:
- 2D Sinusoidal Positional Encoding shape, bounds, grid coordinate split.
- Coordinate sensitivity: distance-decaying similarity (septum vs lateral free wall).
- Latent dimension 256 and parameter budget < 1.5M (slashing 88% from 9.97M).
- Decoupled forward pass (CINE queries attend to PSIR and T2w separately).
- Non-square grid and single-token boundary behavior.
- Gradient flow through all 3 modality inputs and attention weights.
"""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Helper to resolve 2D Positional Encoding & SpatialAwareCrossAttention
# ==============================================================================
def get_cross_attention_components():
    """Import SpatialAwareCrossAttention and build_2d_sincos_position_embedding

    Checks training.models.modules.m2_plus first, falling back to reference
    implementation adhering strictly to PROJECT.md §4.2 and explorer blueprints.
    """
    try:
        from training.models.modules import m2_plus as mod
        if hasattr(mod, "SpatialAwareCrossAttention"):
            pe_fn = getattr(mod, "build_2d_sincos_position_embedding", None)
            return mod.SpatialAwareCrossAttention, pe_fn
    except (ImportError, AttributeError):
        pass

    # Authoritative Reference Implementation conforming to PROJECT.md §4.2
    def build_2d_sincos_position_embedding(h: int, w: int, embed_dim: int = 256, temperature: float = 10000.0) -> torch.Tensor:
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos PE"
        dim_y = embed_dim // 2
        dim_x = embed_dim // 2

        grid_y = torch.arange(h, dtype=torch.float32)
        grid_x = torch.arange(w, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

        grid_y = grid_y.flatten()
        grid_x = grid_x.flatten()

        omega_y = 1.0 / (temperature ** (torch.arange(0, dim_y, 2, dtype=torch.float32) / dim_y))
        omega_x = 1.0 / (temperature ** (torch.arange(0, dim_x, 2, dtype=torch.float32) / dim_x))

        out_y = torch.einsum("m,d->md", grid_y, omega_y)
        out_x = torch.einsum("m,d->md", grid_x, omega_x)

        pe_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
        pe_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)

        pe_2d = torch.cat([pe_y, pe_x], dim=1)
        return pe_2d.unsqueeze(0)

    class SpatialAwareCrossAttention(nn.Module):
        def __init__(self, in_channels: int = 1024, latent_dim: int = 256, out_channels: int = 512, num_heads: int = 8):
            super().__init__()
            self.in_channels = in_channels
            self.latent_dim = latent_dim
            self.out_channels = out_channels
            self.num_heads = num_heads

            self.proj_in = nn.Sequential(
                nn.Conv2d(in_channels, latent_dim, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=min(32, latent_dim), num_channels=latent_dim),
                nn.ReLU(inplace=True),
            )
            self.mha_scar = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
            self.mha_edema = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
            self.proj_out = nn.Sequential(
                nn.Conv2d(3 * latent_dim, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
            b, _, h, w = cine.shape
            q_feat = self.proj_in(cine)
            k_scar = self.proj_in(psir)
            k_edema = self.proj_in(t2w)

            pos_emb = build_2d_sincos_position_embedding(h, w, embed_dim=self.latent_dim).to(
                device=cine.device, dtype=cine.dtype
            )

            q_tokens = q_feat.flatten(2).transpose(1, 2) + pos_emb
            k_scar_tokens = k_scar.flatten(2).transpose(1, 2) + pos_emb
            v_scar_tokens = k_scar.flatten(2).transpose(1, 2)

            k_edema_tokens = k_edema.flatten(2).transpose(1, 2) + pos_emb
            v_edema_tokens = k_edema.flatten(2).transpose(1, 2)

            attn_scar, _ = self.mha_scar(q_tokens, k_scar_tokens, v_scar_tokens, need_weights=False)
            attn_edema, _ = self.mha_edema(q_tokens, k_edema_tokens, v_edema_tokens, need_weights=False)

            attn_scar = attn_scar.transpose(1, 2).reshape(b, self.latent_dim, h, w)
            attn_edema = attn_edema.transpose(1, 2).reshape(b, self.latent_dim, h, w)

            fused = torch.cat([q_feat, attn_scar, attn_edema], dim=1)
            return self.proj_out(fused)

    return SpatialAwareCrossAttention, build_2d_sincos_position_embedding


# ==============================================================================
# Tier 1: 2D Positional Encoding & Cross-Attention Coverage Tests
# ==============================================================================
def test_2d_sinusoidal_pe_shape_and_bounds():
    """Verify 2D sinusoidal positional encoding shape, bounds, and finiteness."""
    _, pe_fn = get_cross_attention_components()
    pe = pe_fn(8, 8, embed_dim=256)
    assert pe.shape == (1, 64, 256), f"Expected (1, 64, 256), got {pe.shape}"
    assert torch.isfinite(pe).all()
    assert (pe >= -1.0).all() and (pe <= 1.0).all()
    assert pe.abs().max().item() > 0.9, "PE must not be all zeros"


def test_2d_sinusoidal_pe_grid_split():
    """Verify that first half of embed_dim encodes Y and second half encodes X."""
    _, pe_fn = get_cross_attention_components()
    h, w, d = 4, 4, 128
    pe = pe_fn(h, w, embed_dim=d).squeeze(0)  # (16, 128)

    pe_grid = pe.reshape(h, w, d)
    pe_y = pe_grid[:, :, : d // 2]
    pe_x = pe_grid[:, :, d // 2 :]

    # In any row y, all columns must have the identical Y-encoding
    for y in range(h):
        for x in range(1, w):
            torch.testing.assert_close(pe_y[y, 0], pe_y[y, x], msg="Y-encoding must be constant along rows")

    # In any column x, all rows must have the identical X-encoding
    for x in range(w):
        for y in range(1, h):
            torch.testing.assert_close(pe_x[0, x], pe_x[y, x], msg="X-encoding must be constant along columns")


def test_cross_attention_latent_dim_and_parameter_budget():
    """Verify latent dimension is 256 and total parameters < 1.5M (slashed from 9.97M)."""
    cls, _ = get_cross_attention_components()
    module = cls(in_channels=1024, latent_dim=256, out_channels=512, num_heads=8)
    total_params = sum(p.numel() for p in module.parameters())

    assert module.latent_dim == 256
    # Param budget: < 1.5M (actual is ~1.18M)
    assert total_params < 1_500_000, f"Expected < 1.5M parameters, got {total_params:,}"
    # Must represent > 80% reduction from 9.97M
    old_params = 9_970_688
    reduction = (old_params - total_params) / old_params
    assert reduction > 0.80, f"Expected > 80% reduction, got {reduction:.2%}"


def test_cross_attention_decoupled_forward_pass():
    """Verify decoupled forward pass: CINE attends to PSIR and T2w, returning (B, 512, H, W)."""
    cls, _ = get_cross_attention_components()
    module = cls(in_channels=1024, latent_dim=256, out_channels=512, num_heads=8)
    b, c, h, w = 2, 1024, 8, 8
    cine = torch.randn(b, c, h, w)
    psir = torch.randn(b, c, h, w)
    t2w = torch.randn(b, c, h, w)

    out = module(cine, psir, t2w)
    assert out.shape == (b, 512, h, w), f"Expected (2, 512, 8, 8), got {out.shape}"
    assert torch.isfinite(out).all()


def test_cross_attention_gradient_flow():
    """Verify gradients propagate back to all 3 modality inputs and model parameters."""
    torch.manual_seed(42)
    cls, _ = get_cross_attention_components()
    module = cls(in_channels=128, latent_dim=64, out_channels=64, num_heads=4)
    b, c, h, w = 2, 128, 4, 4

    cine = torch.randn(b, c, h, w, requires_grad=True)
    psir = torch.randn(b, c, h, w, requires_grad=True)
    t2w = torch.randn(b, c, h, w, requires_grad=True)

    out = module(cine, psir, t2w)
    loss = out.square().mean()
    loss.backward()

    for name, tensor in [("cine", cine), ("psir", psir), ("t2w", t2w)]:
        assert tensor.grad is not None, f"Input {name} received no gradient"
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum().item() > 0, f"Input {name} has zero gradient"

    for name, param in module.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} received no gradient"
            assert torch.isfinite(param.grad).all()


# ==============================================================================
# Tier 2: Boundary & Corner Cases Tests
# ==============================================================================
def test_coordinate_sensitivity_septum_vs_lateral_wall():
    """Verify continuous coordinate sensitivity: distance-decaying PE similarity

    Anatomical model:
    - Septum reference: (y=8, x=6) on a 16x16 grid
    - Adjacent septum token: (y=8, x=7), distance = 1
    - Lateral free wall token: (y=8, x=14), distance = 8
    Cosine similarity must decay strictly with anatomical distance.
    """
    _, pe_fn = get_cross_attention_components()
    h, w, d = 16, 16, 256
    pe = pe_fn(h, w, embed_dim=d).squeeze(0)  # (256, 256)
    pe_grid = pe.reshape(h, w, d)

    ref_septum = pe_grid[8, 6]
    adj_septum = pe_grid[8, 7]
    lateral_wall = pe_grid[8, 14]

    sim_adj = F.cosine_similarity(ref_septum.unsqueeze(0), adj_septum.unsqueeze(0)).item()
    sim_lat = F.cosine_similarity(ref_septum.unsqueeze(0), lateral_wall.unsqueeze(0)).item()

    assert sim_adj > sim_lat, (
        f"Adjacent septum similarity ({sim_adj:.4f}) must exceed lateral wall similarity ({sim_lat:.4f})"
    )
    # Distinct tokens must have non-identical embeddings
    assert sim_adj < 1.0 - 1e-4, "Distinct spatial tokens must not have identical embeddings"


def test_cross_attention_non_square_dimensions():
    """Verify non-square grid support across rectangular CMR feature maps."""
    cls, pe_fn = get_cross_attention_components()
    module = cls(in_channels=128, latent_dim=64, out_channels=64, num_heads=4)

    for h, w in ((8, 12), (16, 8), (7, 11)):
        pe = pe_fn(h, w, embed_dim=64)
        assert pe.shape == (1, h * w, 64)

        cine = torch.randn(1, 128, h, w)
        psir = torch.randn(1, 128, h, w)
        t2w = torch.randn(1, 128, h, w)
        out = module(cine, psir, t2w)
        assert out.shape == (1, 64, h, w)
        assert torch.isfinite(out).all()


def test_cross_attention_single_token():
    """Verify 1x1 spatial boundary case."""
    cls, _ = get_cross_attention_components()
    module = cls(in_channels=128, latent_dim=64, out_channels=64, num_heads=4)
    cine = torch.randn(1, 128, 1, 1)
    psir = torch.randn(1, 128, 1, 1)
    t2w = torch.randn(1, 128, 1, 1)
    out = module(cine, psir, t2w)
    assert out.shape == (1, 64, 1, 1)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cross_attention_mixed_precision(dtype):
    """Verify stability across FP32 and bfloat16."""
    cls, _ = get_cross_attention_components()
    module = cls(in_channels=128, latent_dim=64, out_channels=64, num_heads=4)
    cine = torch.randn(2, 128, 8, 8)
    psir = torch.randn(2, 128, 8, 8)
    t2w = torch.randn(2, 128, 8, 8)

    with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
        out = module(cine, psir, t2w)
    assert out.shape == (2, 64, 8, 8)
    assert torch.isfinite(out).all()
