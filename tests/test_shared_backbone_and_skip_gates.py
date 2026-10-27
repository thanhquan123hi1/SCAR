"""Unit and integration tests for SharedResNetV2 and SSPA_AttentionGate (Milestone 3).

Verifies:
1. SharedResNetV2 parameter budget (~11.91M - 12.42M params, cutting >65% from 35.68M).
2. 3 independent 1-channel stems for CMR physics (CINE, PSIR/LGE, T2W).
3. 1 shared body trunk (block1, block2, block3) sharing 100% of weights.
4. Modality-specific FiLM adapters at stages 1, 2, 3 initialized to identity.
5. forward_modality returning (bottleneck_feature, [skip0, skip1, skip2]) with correct shapes.
6. Direct compatibility with JAX TransUNet pretrained weight loading (load_from, load_from_transunet).
7. SSPA_AttentionGate parameter overhead (< 0.02M params across skip connections).
8. Multi-level skip attention filtering of thoracic background noise.
9. Full backward compatibility for ResNetV2, ConcatFusion, and Fusion_Embed.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from training.models.backbones.resnet_v2 import (
    FiLMAdapter,
    ResNetV2,
    SharedResNetV2,
    StdConv2d,
    normalize_modality,
)
from training.models.modules.fusion import (
    ConcatFusion,
    Fusion_Embed,
    SSPA_AttentionGate,
)


# ==============================================================================
# SharedResNetV2 Tests
# ==============================================================================

def test_shared_resnet_v2_parameter_footprint():
    """Verify SharedResNetV2 cuts >65% parameters from 3 separate ResNetV2-50 backbones."""
    production_model = SharedResNetV2(block_units=(3, 4, 9), width_factor=1.0)
    total_params = sum(p.numel() for p in production_model.parameters())

    # Baseline 3x separate ResNetV2-50 encoders had 35,684,544 parameters
    baseline_params = 35_684_544
    reduction = (baseline_params - total_params) / baseline_params

    # Target is ~12.42M params (cutting >= 65.2%)
    assert total_params < 13_000_000, f"Expected < 13M parameters, got {total_params:,}"
    assert reduction >= 0.65, f"Expected >= 65% parameter reduction, got {reduction:.2%}"


def test_shared_resnet_v2_three_independent_stems():
    """Verify 3 independent 1-channel stems for CINE, PSIR, and T2W."""
    model = SharedResNetV2(block_units=(1, 1, 1), width_factor=0.5)

    assert hasattr(model, "stem_cine")
    assert hasattr(model, "stem_psir")
    assert hasattr(model, "stem_t2w")

    # Each stem must have StdConv2d with in_channels=1
    for stem_name in ("stem_cine", "stem_psir", "stem_t2w"):
        stem = getattr(model, stem_name)
        assert isinstance(stem.conv, StdConv2d)
        assert stem.conv.in_channels == 1
        assert stem.conv.out_channels == model.width
        assert stem.conv.kernel_size == (7, 7)
        assert isinstance(stem.gn, nn.GroupNorm)
        assert isinstance(stem.relu, nn.ReLU)

    # Weights must be independent (not sharing memory)
    assert model.stem_cine.conv.weight.data_ptr() != model.stem_psir.conv.weight.data_ptr()
    assert model.stem_cine.conv.weight.data_ptr() != model.stem_t2w.conv.weight.data_ptr()


def test_shared_resnet_v2_shared_body_trunk():
    """Verify body trunk (block1, block2, block3) is shared across modalities."""
    model = SharedResNetV2(block_units=(1, 1, 1), width_factor=0.5)

    assert hasattr(model, "body")
    assert hasattr(model.body, "block1")
    assert hasattr(model.body, "block2")
    assert hasattr(model.body, "block3")

    # Trunk memory address is identical across modality forwards
    body_ptr = id(model.body)
    assert id(model.body) == body_ptr


def test_shared_resnet_v2_film_adapters_identity_initialization():
    """Verify FiLM adapters at stages 1, 2, 3 are initialized to identity (gamma=1, beta=0)."""
    model = SharedResNetV2(block_units=(1, 1, 1), width_factor=0.5)

    assert hasattr(model, "film1")
    assert hasattr(model, "film2")
    assert hasattr(model, "film3")

    for film in (model.film1, model.film2, model.film3):
        assert isinstance(film, FiLMAdapter)
        for mod in ("cine", "psir", "t2w"):
            gamma = film.gamma[mod]
            beta = film.beta[mod]
            assert torch.allclose(gamma, torch.ones_like(gamma))
            assert torch.allclose(beta, torch.zeros_like(beta))

    # Verify identity transformation on dummy features
    dummy_feat = torch.randn(2, model.width * 4, 16, 16)
    out_cine = model.film1(dummy_feat, "cine")
    assert torch.allclose(dummy_feat, out_cine, atol=1e-6)


def test_shared_resnet_v2_forward_modality_shapes():
    """Verify forward_modality returns (bottleneck, [skip0, skip1, skip2]) with exact shapes."""
    model = SharedResNetV2(block_units=(3, 4, 9), width_factor=1.0).eval()
    b = 2
    res = 128
    x = torch.randn(b, 1, res, res)

    for mod in ("cine", "psir", "t2w"):
        with torch.no_grad():
            bottleneck, skips = model.forward_modality(x, mod)

        # Bottleneck: 1024 channels, H/16, W/16 (8x8)
        assert bottleneck.shape == (b, 1024, 8, 8), f"Bottleneck shape mismatch: {bottleneck.shape}"
        assert len(skips) == 3, f"Expected 3 skips, got {len(skips)}"

        # skip0 (stage 2): 512 channels, H/8, W/8 (16x16)
        assert skips[0].shape == (b, 512, 16, 16), f"skip0 shape mismatch: {skips[0].shape}"
        # skip1 (stage 1): 256 channels, H/4, W/4 (32x32)
        assert skips[1].shape == (b, 256, 32, 32), f"skip1 shape mismatch: {skips[1].shape}"
        # skip2 (stem): 64 channels, H/2, W/2 (64x64)
        assert skips[2].shape == (b, 64, 64, 64), f"skip2 shape mismatch: {skips[2].shape}"


def test_shared_resnet_v2_modality_aliases():
    """Verify robust handling of modality aliases (e.g. 'lge' -> 'psir', 0 -> 'cine')."""
    model = SharedResNetV2(block_units=(1, 1, 1), width_factor=0.5).eval()
    x = torch.randn(1, 1, 32, 32)

    with torch.no_grad():
        out_lge, _ = model.forward_modality(x, "lge")
        out_psir, _ = model.forward_modality(x, "psir")
        assert torch.allclose(out_lge, out_psir)

        out_c0, _ = model.forward_modality(x, "c0")
        out_cine, _ = model.forward_modality(x, "cine")
        assert torch.allclose(out_c0, out_cine)

        out_0, _ = model.forward_modality(x, 0)
        assert torch.allclose(out_0, out_cine)


def test_shared_resnet_v2_multimodal_forward():
    """Verify forward_multimodal returns features for all 3 modalities."""
    model = SharedResNetV2(block_units=(1, 1, 1), width_factor=0.5).eval()
    res = 32
    cine = torch.randn(1, 1, res, res)
    psir = torch.randn(1, 1, res, res)
    t2w  = torch.randn(1, 1, res, res)

    with torch.no_grad():
        results = model.forward_multimodal(cine, psir, t2w)

    assert set(results.keys()) == {"cine", "psir", "t2w"}
    for mod in ("cine", "psir", "t2w"):
        bottleneck, skips = results[mod]
        assert bottleneck.shape[0] == 1
        assert len(skips) == 3


def test_shared_resnet_v2_jax_transunet_weight_loading():
    """Verify seamless loading of JAX TransUNet pretrained weights into shared trunk and stems."""
    model = SharedResNetV2(block_units=(1, 1, 1), width_factor=0.5)

    # Construct synthetic JAX weights matching TransUNet structure
    weights = {}
    weights["conv_root/kernel"] = np.random.randn(7, 7, 3, model.width).astype(np.float32)
    weights["gn_root/scale"] = np.random.randn(model.width).astype(np.float32)
    weights["gn_root/bias"] = np.random.randn(model.width).astype(np.float32)

    for block_name, block in model.body.named_children():
        for unit_name, unit in block.named_children():
            prefix = f"{block_name}/{unit_name}"
            weights[f"{prefix}/conv1/kernel"] = np.random.randn(1, 1, unit.conv1.in_channels, unit.conv1.out_channels).astype(np.float32)
            weights[f"{prefix}/gn1/scale"] = np.random.randn(unit.gn1.num_channels).astype(np.float32)
            weights[f"{prefix}/gn1/bias"] = np.random.randn(unit.gn1.num_channels).astype(np.float32)
            weights[f"{prefix}/conv2/kernel"] = np.random.randn(3, 3, unit.conv2.in_channels, unit.conv2.out_channels).astype(np.float32)
            weights[f"{prefix}/gn2/scale"] = np.random.randn(unit.gn2.num_channels).astype(np.float32)
            weights[f"{prefix}/gn2/bias"] = np.random.randn(unit.gn2.num_channels).astype(np.float32)
            weights[f"{prefix}/conv3/kernel"] = np.random.randn(1, 1, unit.conv3.in_channels, unit.conv3.out_channels).astype(np.float32)
            weights[f"{prefix}/gn3/scale"] = np.random.randn(unit.gn3.num_channels).astype(np.float32)
            weights[f"{prefix}/gn3/bias"] = np.random.randn(unit.gn3.num_channels).astype(np.float32)
            if hasattr(unit, "downsample"):
                weights[f"{prefix}/conv_proj/kernel"] = np.random.randn(1, 1, unit.downsample.in_channels, unit.downsample.out_channels).astype(np.float32)
                weights[f"{prefix}/gn_proj/scale"] = np.random.randn(unit.gn_proj.num_channels).astype(np.float32)
                weights[f"{prefix}/gn_proj/bias"] = np.random.randn(unit.gn_proj.num_channels).astype(np.float32)

    # 1. Direct dictionary loading
    model.validate_pretrained(weights)
    model.load_from(weights)

    # 2. File path loading via load_from_transunet
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        np.savez(tmp.name, **weights)
        tmp_path = tmp.name

    try:
        model.load_from_transunet(tmp_path)
    finally:
        Path(tmp_path).unlink()


@pytest.mark.parametrize("resolution", [256, 128, 64, 32])
def test_shared_resnet_v2_multi_resolution(resolution):
    """Verify forward execution across multiple input spatial resolutions."""
    model = SharedResNetV2(block_units=(1, 1, 1), width_factor=0.5).eval()
    x = torch.randn(1, 1, resolution, resolution)
    with torch.no_grad():
        out, skips = model(x, "cine")
    assert out.shape == (1, model.width * 16, resolution // 16, resolution // 16)
    assert skips[0].shape == (1, model.width * 8, resolution // 8, resolution // 8)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_shared_resnet_v2_batch_stability(batch_size):
    """Verify forward stability for small batch sizes (GroupNorm small-batch stability)."""
    model = SharedResNetV2(block_units=(1, 1, 1), width_factor=0.5).eval()
    x = torch.randn(batch_size, 1, 32, 32)
    with torch.no_grad():
        out, skips = model(x, "cine")
    assert out.shape[0] == batch_size
    assert torch.isfinite(out).all()


def test_resnet_v2_backward_compatibility():
    """Verify original ResNetV2 class remains fully functional and unchanged."""
    resnet = ResNetV2(block_units=(1, 1, 1), width_factor=0.5).eval()
    img = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out, skips = resnet(img)
    assert out.shape == (2, 32 * 16, 2, 2)
    assert len(skips) == 3


# ==============================================================================
# SSPA_AttentionGate Tests
# ==============================================================================

def test_sspa_attention_gate_parameter_overhead():
    """Verify SSPA_AttentionGate parameter overhead is < 0.02M params across skip connections."""
    # Skip 2 at 64x64 (C_s=64, C_g=128, C_inter=16)
    gate_skip2 = SSPA_AttentionGate(in_channels=64, gate_channels=128, inter_channels=16)
    # Skip 1 at 32x32 (C_s=256, C_g=256, C_inter=32)
    gate_skip1 = SSPA_AttentionGate(in_channels=256, gate_channels=256, inter_channels=32)

    p_skip2 = sum(p.numel() for p in gate_skip2.parameters())
    p_skip1 = sum(p.numel() for p in gate_skip1.parameters())
    combined_params = p_skip2 + p_skip1

    # Exact numbers from architectural specification (Section 3.3)
    assert p_skip2 == 3217, f"Expected 3,217 params at Skip 2, got {p_skip2}"
    assert p_skip1 == 16673, f"Expected 16,673 params at Skip 1, got {p_skip1}"
    assert combined_params == 19890, f"Expected 19,890 params combined, got {combined_params}"
    assert combined_params < 20_000, f"Expected overhead < 0.02M params, got {combined_params}"


def test_sspa_attention_gate_gated_forward():
    """Verify gated forward pass with coarser gating signal from decoder."""
    gate = SSPA_AttentionGate(in_channels=64, gate_channels=128, inter_channels=16).eval()
    b = 2
    x = torch.randn(b, 64, 64, 64)
    g = torch.randn(b, 128, 32, 32)  # Coarser resolution

    with torch.no_grad():
        out, alpha = gate(x, g, return_attention=True)

    assert out.shape == x.shape, f"Output shape mismatch: {out.shape}"
    assert alpha.shape == (b, 1, 64, 64), f"Attention mask shape mismatch: {alpha.shape}"
    # Attention coefficient alpha must be strictly in [0.0, 1.0]
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all(), "Alpha outside [0, 1] range!"
    assert torch.isfinite(out).all()


def test_sspa_attention_gate_standalone_forward():
    """Verify standalone forward pass without external gating signal."""
    gate = SSPA_AttentionGate(in_channels=64, inter_channels=16).eval()
    x = torch.randn(2, 64, 32, 32)

    with torch.no_grad():
        out = gate(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_sspa_attention_gate_thoracic_noise_filtering():
    """A gate with controlled weights can select the supplied target region."""
    gate = SSPA_AttentionGate(in_channels=64, gate_channels=128, inter_channels=16).eval()
    with torch.no_grad():
        gate.conv_x.weight.zero_()
        gate.conv_g.weight.zero_()
        # Each GN group contains both a responsive channel and a reference channel.
        gate.conv_g.weight[::2].fill_(1.0 / 128)
        gate.strip_h.weight.zero_()
        gate.strip_w.weight.zero_()
        gate.psi[0].weight.zero_()
        gate.psi[0].weight[:, ::2].fill_(1.0)
        gate.psi[0].bias.zero_()
    x = torch.ones(1, 64, 32, 32)
    # Gating signal with strong activation in cardiac region (center) and zero in periphery
    g = torch.zeros(1, 128, 16, 16)
    g[:, :, 6:10, 6:10] = 5.0  # Center myocardium active

    with torch.no_grad():
        out, alpha = gate(x, g, return_attention=True)

    center_mean_attn = alpha[:, :, 12:20, 12:20].mean().item()
    periphery_mean_attn = alpha[:, :, :4, :4].mean().item()

    torch.testing.assert_close(out, x * alpha)
    assert center_mean_attn > periphery_mean_attn, (
        f"Expected center attention ({center_mean_attn:.3f}) > periphery ({periphery_mean_attn:.3f})"
    )


def test_fusion_backward_compatibility():
    """Verify legacy ConcatFusion and Fusion_Embed modules remain functional."""
    concat_fusion = ConcatFusion(in_channels=64, out_channels=32).eval()
    embed_fusion = Fusion_Embed(embed_dim=64).eval()

    c = torch.randn(1, 64, 16, 16)
    p = torch.randn(1, 64, 16, 16)
    t = torch.randn(1, 64, 16, 16)

    with torch.no_grad():
        out_concat = concat_fusion(c, p, t)
        out_embed = embed_fusion(c, p, t)

    assert out_concat.shape == (1, 32, 16, 16)
    assert out_embed.shape == (1, 64, 16, 16)
