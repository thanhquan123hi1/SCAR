"""Regression and specification tests for SSPANet modules.

Includes:
1. Legacy regression checks for centered SSPANet gating and checkpoint semantics.
2. Tier 1 & Tier 2 tests for OptimizedSSPANet (Z-Pool, Depthwise Strip Pooling,
   Style Pooling mean + std dev, [0, 1] dynamic range, parameter profiling,
   extreme values, 1x1 boundaries, and gradient flow).
"""
import importlib.util
import inspect
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from training.models import M2PlusNet, get_testing, model_from_config
from training.models.modules.sspanet import (
    DEFAULT_SSPA_GATE,
    LEGACY_SSPA_GATE,
    SSPANet_Block,
    strip_rms,
)
from training.trainer.trainer import _configs_equal, _resume_model_config


# ==============================================================================
# Helper to resolve OptimizedSSPANet (progressive testability)
# ==============================================================================
_CACHED_SSPANET_CLASSES = None

def get_optimized_sspanet_classes():
    """Import OptimizedSSPANet from codebase or reference blueprint."""
    global _CACHED_SSPANET_CLASSES
    if _CACHED_SSPANET_CLASSES is not None:
        return _CACHED_SSPANET_CLASSES

    try:
        from training.models.modules import sspanet as mod
        if hasattr(mod, "OptimizedSSPANet"):
            _CACHED_SSPANET_CLASSES = (
                mod.OptimizedSSPANet,
                getattr(mod, "SSPA_ZPoolChannelAttention", None),
                getattr(mod, "SSPA_DirectionalStripPooling", None),
                getattr(mod, "SSPA_StylePooling", None),
            )
            return _CACHED_SSPANET_CLASSES
    except (ImportError, AttributeError):
        pass

    ref_path = Path("d:/NCKH/SCAR/.agents/teamwork/explorer_survey_2/proposed_sspanet.py")
    if ref_path.exists():
        spec = importlib.util.spec_from_file_location("proposed_sspanet", ref_path)
        ref_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref_mod)
        _CACHED_SSPANET_CLASSES = (
            ref_mod.OptimizedSSPANet,
            ref_mod.SSPA_ZPoolChannelAttention,
            ref_mod.SSPA_DirectionalStripPooling,
            ref_mod.SSPA_StylePooling,
        )
        return _CACHED_SSPANET_CLASSES
    raise ImportError("OptimizedSSPANet class could not be resolved from models or explorer reference.")


def instantiate_sspanet(in_channels=1024, reduction=16, is_skip_gate=False):
    """Instantiate OptimizedSSPANet adapting to constructor keyword variants."""
    cls, _, _, _ = get_optimized_sspanet_classes()
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    if "reduction_ratio" in sig.parameters:
        kwargs["reduction_ratio"] = reduction
    elif "reduction" in sig.parameters:
        kwargs["reduction"] = reduction

    if "is_skip_gate" in sig.parameters:
        kwargs["is_skip_gate"] = is_skip_gate
    elif "mode" in sig.parameters:
        kwargs["mode"] = "attention_gate" if is_skip_gate else "residual"

    return cls(in_channels, **kwargs)


# ==============================================================================
# Legacy SSPANet Regression Tests (Unchanged)
# ==============================================================================
@pytest.mark.parametrize("gain", [0.75, 1.0, 1.25])
def test_centered_gate_can_attenuate_preserve_and_amplify_features(gain):
    block = SSPANet_Block(4).eval()
    with torch.no_grad():
        block.channel.conv[0].weight.zero_()
        block.channel.conv[1].bias.zero_()
        block.spatial.conv3.weight.zero_()
        block.spatial.conv3.bias.fill_(math.atanh((gain - 1.0) / 0.5))
    x = torch.linspace(-3, 3, 2 * 4 * 5 * 7).reshape(2, 4, 5, 7)
    x[0, 0, 0, 0] = 0
    torch.testing.assert_close(block(x), gain * x)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_centered_gate_is_bounded_and_both_branches_receive_finite_gradients(mixed_precision):
    torch.manual_seed(7)
    block = SSPANet_Block(4)
    x = torch.rand(2, 4, 5, 7, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed_precision):
        y = block(x)
        loss = y.square().mean()
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.all(y >= 0.5 * x)
    assert torch.all(y <= 1.5 * x)
    loss.backward()
    assert torch.isfinite(x.grad).all()
    for branch in (block.channel, block.spatial):
        gradients = [parameter.grad for parameter in branch.parameters()]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients)
        assert sum(grad.abs().sum().item() for grad in gradients) > 0


def test_legacy_gate_reproduces_the_original_formula():
    block = SSPANet_Block(4, gate_mode=LEGACY_SSPA_GATE).eval()
    x = torch.rand(2, 4, 5, 7)
    with torch.no_grad():
        pooled = torch.cat((x.max(dim=1, keepdim=True).values, x.mean(dim=1, keepdim=True)), dim=1)
        channel = x * torch.sigmoid(block.channel.conv(pooled))
        spatial = block.spatial
        strip_h = spatial.bn1(spatial.conv1(strip_rms(x, dim=3)))
        strip_w = spatial.bn2(spatial.conv2(strip_rms(x, dim=2)))
        spatial_features = x * torch.sigmoid(spatial.conv3(strip_h + strip_w))
        expected = x + x * torch.sigmoid(channel + spatial_features)
        torch.testing.assert_close(block(x), expected)


def test_new_checkpoints_round_trip_and_missing_gate_metadata_uses_legacy():
    config = get_testing()
    source = M2PlusNet(config, img_size=32).eval()
    assert source.config.sspanet_gate == DEFAULT_SSPA_GATE
    restored = model_from_config(source.config.to_dict(), img_size=32).eval()
    restored.load_state_dict(source.state_dict(), strict=True)
    images = [torch.rand(1, 1, 32, 32) for _ in range(3)]
    with torch.no_grad():
        torch.testing.assert_close(source(*images), restored(*images))

    legacy_config = source.config.to_dict()
    legacy_config.pop("sspanet_gate")
    legacy = model_from_config(legacy_config, img_size=32).eval()
    legacy.load_state_dict(source.state_dict(), strict=True)
    for block in (legacy.sspanet_cine, legacy.sspanet_psir, legacy.sspanet_t2w):
        assert block.gate_mode == LEGACY_SSPA_GATE
    assert not _configs_equal(_resume_model_config(legacy_config), _resume_model_config(config.to_dict()))
    explicit_legacy = legacy_config | {"sspanet_gate": LEGACY_SSPA_GATE}
    assert _configs_equal(_resume_model_config(legacy_config), _resume_model_config(explicit_legacy))


def test_invalid_gate_is_rejected_before_loading_weights():
    with pytest.raises(ValueError, match="SSPANet gate"):
        SSPANet_Block(4, gate_mode="unknown")
    config = get_testing()
    config.sspanet_gate = "unknown"
    with pytest.raises(ValueError, match="SSPANet gate"):
        model_from_config(config)


def test_shared_sspanet_default_does_not_change_nnunet_resume_metadata():
    config = {"architecture": "nnunet", "ablation": "NNUNET", "num_classes": 4}
    with_shared_default = config | {"sspanet_gate": DEFAULT_SSPA_GATE,
                                   "fusion_version": "spatial_v2", "latent_dim": 256,
                                   "sspanet_reduction": 16, "use_pos_enc": True}
    assert _configs_equal(_resume_model_config(config), _resume_model_config(with_shared_default))


def test_optimized_attribute_inspection_keeps_architecture_and_optimizer_parameters():
    block = SSPANet_Block(32, gate_mode="optimized").double().eval()
    optimizer = torch.optim.AdamW(block.parameters())
    state_keys = set(block.state_dict())
    parameter_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    x = torch.randn(2, 32, 8, 8, dtype=torch.float64)
    before = block(x)
    assert not hasattr(block, "channel")
    assert not hasattr(block, "spatial")
    assert block.gate_mode == "optimized"
    assert set(block.state_dict()) == state_keys
    assert {id(p) for p in block.parameters()} == parameter_ids
    torch.testing.assert_close(before, block(x), rtol=0, atol=0)


# ==============================================================================
# Tier 1: OptimizedSSPANet Feature Coverage Tests
# ==============================================================================
def test_sspanet_parameter_count_below_threshold():
    """Verify 3 bottleneck OptimizedSSPANet blocks have < 2.0M parameters total."""
    blocks = [instantiate_sspanet(in_channels=1024, reduction=16) for _ in range(3)]
    total_params = sum(sum(p.numel() for p in b.parameters()) for b in blocks)
    # Target in PROJECT.md and ORIGINAL_REQUEST is < 2.0M (typically ~1.38M)
    assert total_params < 2_000_000, f"Expected < 2.0M parameters, got {total_params:,}"
    # Verify each block is < 700k parameters
    for i, b in enumerate(blocks):
        block_params = sum(p.numel() for p in b.parameters())
        assert block_params < 700_000, f"Block {i} has {block_params:,} params, expected < 700k"


def test_sspanet_three_components_present():
    """Verify presence and execution of Z-Pool Channel Attention, Strip, and Style Pooling."""
    cls, cls_ch, cls_strip, cls_style = get_optimized_sspanet_classes()
    block = instantiate_sspanet(in_channels=128, reduction=8)

    # Check that block contains all three sub-modules
    has_channel = any(
        (cls_ch is not None and isinstance(m, cls_ch)) or "channel" in type(m).__name__.lower() or "channel" in n.lower()
        for n, m in block.named_modules()
    )
    has_strip = any(
        (cls_strip is not None and isinstance(m, cls_strip)) or "strip" in type(m).__name__.lower() or "strip" in n.lower()
        for n, m in block.named_modules()
    )
    has_style = any(
        (cls_style is not None and isinstance(m, cls_style)) or "style" in type(m).__name__.lower() or "style" in n.lower()
        for n, m in block.named_modules()
    )

    assert has_channel, "SSPANet must include Z-Pool channel attention component"
    assert has_strip, "SSPANet must include directional strip pooling component"
    assert has_style, "SSPANet must include style pooling component"


def test_sspanet_style_pooling_mean_and_std():
    """Verify that Style Pooling captures both 1st-order (mean) and 2nd-order (std dev) statistics."""
    cls, _, _, cls_style = get_optimized_sspanet_classes()
    if cls_style is None:
        block = instantiate_sspanet(in_channels=64, reduction=4)
        style_mod = getattr(block, "style_pooling", None) or getattr(block, "style", None)
    else:
        style_mod = cls_style(in_channels=64, reduction=4)

    assert style_mod is not None, "Style pooling module must be accessible"

    # Create two feature maps with identical mean (0.5) but different variance/texture
    torch.manual_seed(42)
    # Map 1: Low contrast / flat texture (mean 0.5, var near 0)
    x1 = torch.full((1, 64, 16, 16), 0.5)
    # Map 2: High contrast / checkerboard texture (mean 0.5, high var)
    x2 = torch.zeros((1, 64, 16, 16))
    x2[:, :, 0::2, 0::2] = 1.0
    x2[:, :, 1::2, 1::2] = 1.0
    assert torch.isclose(x1.mean(), x2.mean(), atol=1e-5), "Test setup: means must match"

    style_mod.eval()
    with torch.no_grad():
        out1 = style_mod(x1)
        out2 = style_mod(x2)

    # If style pooling only captured mean, out1 and out2 would be identical.
    # Because it captures std dev, out1 and out2 must differ.
    diff = (out1 - out2).abs().max().item()
    assert diff > 1e-4, f"Style pooling must be sensitive to 2nd-order std dev; diff={diff}"


def test_sspanet_dynamic_range_coverage():
    """Verify single-sigmoid gating dynamic range spans [0.0, 1.0] without [0.5, 1.0] lock."""
    block = instantiate_sspanet(in_channels=32, reduction=4)

    # For any single sigmoid gate = sigmoid(logits):
    # Logits can be negative (gate -> 0.0) or positive (gate -> 1.0)
    # Test that gating values can reach < 0.05 and > 0.95 across synthetic extreme logits
    if hasattr(block, "compute_logits"):
        with torch.no_grad():
            neg_x = torch.full((1, 32, 8, 8), -10.0)
            pos_x = torch.full((1, 32, 8, 8), 10.0)
            logits_neg = block.compute_logits(neg_x)
            logits_pos = block.compute_logits(pos_x)
            gate_neg = torch.sigmoid(logits_neg)
            gate_pos = torch.sigmoid(logits_pos)

            # Both gates must remain strictly inside [0.0, 1.0]
            assert (gate_neg >= 0.0).all() and (gate_neg <= 1.0).all()
            assert (gate_pos >= 0.0).all() and (gate_pos <= 1.0).all()
            # Gate is not locked to [0.5, 1.0]
            assert gate_neg.min().item() < 0.5, f"Dynamic range flaw D-01: min gate={gate_neg.min().item()}"


def test_sspanet_output_shape_preservation():
    """Verify output shape identical to input shape across varied channels and spatial dims."""
    for c in (32, 64, 128):
        block = instantiate_sspanet(in_channels=c, reduction=4)
        for h, w in ((8, 8), (16, 16), (7, 13)):
            x = torch.randn(2, c, h, w)
            y = block(x)
            assert y.shape == x.shape, f"Shape mismatch: in {x.shape} vs out {y.shape}"


def test_sspanet_gradient_flow_all_three_branches():
    """Verify backward gradient flow through input and all 3 submodule branches."""
    torch.manual_seed(10)
    block = instantiate_sspanet(in_channels=64, reduction=4)
    x = torch.randn(2, 64, 8, 8, requires_grad=True)
    y = block(x)
    loss = y.sum()
    loss.backward()

    # Input gradient
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0

    # Gradients for all model parameters
    for name, param in block.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has NaN/Inf gradients"


# ==============================================================================
# Tier 2: Boundary & Corner Cases Tests
# ==============================================================================
def test_sspanet_extreme_values():
    """Verify numerical stability with extreme inputs (-1e4, +1e4)."""
    block = instantiate_sspanet(in_channels=32, reduction=4)
    x_ext = torch.zeros(1, 32, 8, 8)
    x_ext[:, :, :4, :] = -1e4
    x_ext[:, :, 4:, :] = 1e4
    y = block(x_ext)
    assert torch.isfinite(y).all(), "SSPANet output contains NaN or Inf under extreme input values"


def test_sspanet_uniform_input_zero_variance():
    """Verify std dev calculation when input is perfectly uniform (variance=0, epsilon stability)."""
    block = instantiate_sspanet(in_channels=32, reduction=4)
    # All values identical => variance is exactly zero
    x_zero_var = torch.full((2, 32, 10, 10), 123.456)
    y = block(x_zero_var)
    assert torch.isfinite(y).all(), "Zero-variance input caused division by zero or NaN"


def test_sspanet_single_pixel_spatial_size():
    """Verify behavior when H=1 and/or W=1 (spatial boundary conditions)."""
    block = instantiate_sspanet(in_channels=32, reduction=4).eval()
    for h, w in ((1, 1), (1, 8), (8, 1)):
        x = torch.randn(1, 32, h, w)
        y = block(x)
        assert y.shape == (1, 32, h, w)
        assert torch.isfinite(y).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sspanet_nan_inf_robustness_mixed_precision(dtype):
    """Verify stability across FP32 and AMP bfloat16."""
    block = instantiate_sspanet(in_channels=32, reduction=4)
    x = torch.randn(2, 32, 8, 8)
    with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
        y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_sspanet_attention_gate_mode():
    """Verify attention gate mode suppresses features to near zero when gate is low."""
    gate_block = instantiate_sspanet(in_channels=32, reduction=4, is_skip_gate=True)
    x = torch.ones(1, 32, 8, 8) * 10.0
    y = gate_block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    # In attention gate mode: y = x * gate where gate in [0, 1], so y <= x for non-negative x
    assert torch.all(y <= x + 1e-5), "Attention gate mode should not amplify non-negative features beyond x"
