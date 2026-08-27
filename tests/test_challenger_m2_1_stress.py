"""Comprehensive Adversarial Stress Test Suite for Milestone 2.

Role: Challenger M2_1 (2D Model Stress & Resolution Challenger)
Target Models:
- ResUNetPlusPlus2D
- MultiDatasetResUNetPlusPlus
- ResUNetPlusPlusEncoder
- ResUNetPlusPlusDecoder
- AttentionBlock, ASPP, ResidualConv, Stem_Block
"""

import sys
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from training.models.resunet_plus_plus import (
    ResUNetPlusPlus2D,
    MultiDatasetResUNetPlusPlus,
    ResUNetPlusPlusEncoder,
    ResUNetPlusPlusDecoder,
)
from training.models.modules import (
    AttentionBlock,
    ASPP,
    ResidualConv,
    Stem_Block,
    Squeeze_Excite_Block,
)


def test_encoder_five_stages():
    """Verify ResUNetPlusPlusEncoder returns all 5 hierarchical stages."""
    print("--- Test 1: Encoder 5-stage Hierarchy ---")
    encoder = ResUNetPlusPlusEncoder(in_channels=3, filters=[32, 64, 128, 256, 512])
    encoder.eval()
    
    test_shapes = [
        (2, 3, 256, 256),
        (2, 3, 219, 219),
        (1, 3, 225, 225),
        (2, 3, 191, 203),
        (1, 3, 257, 255),
    ]
    
    for shape in test_shapes:
        x = torch.randn(*shape)
        x1, x2, x3, x4, x5 = encoder(x)
        
        # Verify stage channel counts
        assert x1.shape[1] == 32, f"x1 channels mismatch: {x1.shape}"
        assert x2.shape[1] == 64, f"x2 channels mismatch: {x2.shape}"
        assert x3.shape[1] == 128, f"x3 channels mismatch: {x3.shape}"
        assert x4.shape[1] == 256, f"x4 channels mismatch: {x4.shape}"
        assert x5.shape[1] == 512, f"x5 channels mismatch: {x5.shape}"
        
        # Verify stage spatial dimensions
        assert x1.shape[2:] == (shape[2], shape[3]), f"x1 spatial mismatch: {x1.shape}"
        print(f"  Shape {shape} -> x1:{list(x1.shape)}, x2:{list(x2.shape)}, x3:{list(x3.shape)}, x4:{list(x4.shape)}, x5:{list(x5.shape)}")

    print("  [PASS] Encoder 5-stage hierarchy verified.\n")


def test_arbitrary_odd_resolutions_resunet2d():
    """Stress test ResUNetPlusPlus2D across diverse, odd, non-square, and extreme resolutions."""
    print("--- Test 2: ResUNetPlusPlus2D Diverse & Odd Resolutions ---")
    
    resolutions = [
        # Required in prompt
        (2, 3, 219, 219),
        (1, 3, 225, 225),
        (2, 3, 191, 203),
        (1, 3, 257, 255),
        # Boundary / Prime / Non-square
        (1, 3, 197, 211),
        (2, 3, 131, 149),
        (1, 3, 64, 64),
        (2, 3, 65, 67),
        (1, 3, 48, 48),
        (1, 3, 33, 33),
        (1, 3, 96, 256),
        (2, 3, 256, 96),
        (1, 3, 127, 311),
        (1, 3, 384, 512),
    ]
    
    for one_vs_rest in [False, True]:
        num_classes = 5
        expected_out_c = 4 if one_vs_rest else 5
        model = ResUNetPlusPlus2D(in_channels=3, num_classes=num_classes, one_vs_rest=one_vs_rest)
        model.eval()
        
        for shape in resolutions:
            x = torch.randn(*shape)
            out = model(x)
            expected_shape = (shape[0], expected_out_c, shape[2], shape[3])
            assert out.shape == expected_shape, f"Resolution {shape} (OVR={one_vs_rest}) output shape mismatch: {out.shape} vs {expected_shape}"
            assert not torch.isnan(out).any(), f"NaN in output for shape {shape}"
            assert not torch.isinf(out).any(), f"Inf in output for shape {shape}"
            print(f"  OVR={one_vs_rest} | In: {shape} -> Out: {list(out.shape)} | NaNs: 0, Infs: 0 [PASS]")

    print("  [PASS] ResUNetPlusPlus2D arbitrary odd resolutions verified.\n")


def test_multidataset_resunet_plus_plus():
    """Stress test MultiDatasetResUNetPlusPlus across view decoders and resolutions."""
    print("--- Test 3: MultiDatasetResUNetPlusPlus Multi-View & Error Handling ---")
    
    multi_model = MultiDatasetResUNetPlusPlus(
        in_channels=3,
        num_classes_map={"2ch": 4, "4ch": 5, "sa": 5, "ras154": 2},
        one_vs_rest=True
    )
    multi_model.eval()
    
    expected_channels = {
        "2ch": 3,   # 4 - 1
        "4ch": 4,   # 5 - 1
        "sa": 4,    # 5 - 1
        "ras154": 1 # 2 - 1
    }
    
    test_shapes = [
        (2, 3, 219, 219),
        (1, 3, 225, 225),
        (2, 3, 191, 203),
        (1, 3, 257, 255),
    ]
    
    for view, exp_c in expected_channels.items():
        for shape in test_shapes:
            x = torch.randn(*shape)
            out = multi_model(x, dataset_type=view)
            expected_shape = (shape[0], exp_c, shape[2], shape[3])
            assert out.shape == expected_shape, f"View {view} shape {shape} mismatch: {out.shape} vs {expected_shape}"
            assert not torch.isnan(out).any(), f"NaN in view {view} output for shape {shape}"
            assert not torch.isinf(out).any(), f"Inf in view {view} output for shape {shape}"
        print(f"  View '{view}' (exp channels={exp_c}) across all odd resolutions: PASSED")
        
    # Test invalid view error handling
    try:
        x = torch.randn(1, 3, 64, 64)
        multi_model(x, dataset_type="invalid_view_name")
        assert False, "Should have raised ValueError for invalid dataset_type"
    except ValueError as e:
        print(f"  Invalid view name successfully caught: {e}")
        
    print("  [PASS] MultiDatasetResUNetPlusPlus verified.\n")


def test_full_backward_pass_all_parameters():
    """Verify full backward pass produces non-zero, non-NaN gradients for 100% of parameters."""
    print("--- Test 4: Full Backward Pass Gradient Flow Verification ---")
    
    # 1. ResUNetPlusPlus2D
    torch.manual_seed(42)
    model = ResUNetPlusPlus2D(in_channels=3, num_classes=5, one_vs_rest=False)
    model.train()
    
    shapes_to_test = [
        (2, 3, 219, 219),
        (1, 3, 225, 225),
        (2, 3, 191, 203),
        (1, 3, 257, 255),
    ]
    
    for shape in shapes_to_test:
        model.zero_grad()
        x = torch.randn(*shape, requires_grad=True)
        target = torch.randn(shape[0], 5, shape[2], shape[3])
        
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        
        # Check input gradient
        assert x.grad is not None, f"Input tensor x.grad is None for shape {shape}"
        assert not torch.isnan(x.grad).any(), f"Input grad contains NaN for shape {shape}"
        assert not torch.isinf(x.grad).any(), f"Input grad contains Inf for shape {shape}"
        assert x.grad.abs().sum() > 0, f"Input grad is completely zero for shape {shape}"
        
        # Check every named parameter
        zero_grad_params = []
        nan_grad_params = []
        inf_grad_params = []
        total_params = 0
        total_param_elements = 0
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            total_params += 1
            total_param_elements += param.numel()
            
            if param.grad is None:
                zero_grad_params.append((name, "grad is None"))
            elif torch.isnan(param.grad).any():
                nan_grad_params.append(name)
            elif torch.isinf(param.grad).any():
                inf_grad_params.append(name)
            elif param.grad.abs().sum() == 0:
                zero_grad_params.append((name, "grad sum == 0"))
                
        assert len(nan_grad_params) == 0, f"Parameters with NaN grad: {nan_grad_params}"
        assert len(inf_grad_params) == 0, f"Parameters with Inf grad: {inf_grad_params}"
        assert len(zero_grad_params) == 0, f"Parameters with zero grad: {zero_grad_params}"
        
        print(f"  Shape {shape}: 100% parameter gradients active! ({total_params} parameter tensors, {total_param_elements:,} weights) [PASS]")

    # 2. MultiDatasetResUNetPlusPlus backward pass on each head
    torch.manual_seed(42)
    multi_model = MultiDatasetResUNetPlusPlus(
        in_channels=3,
        num_classes_map={"2ch": 4, "4ch": 5, "sa": 5, "ras154": 2},
        one_vs_rest=True
    )
    multi_model.train()
    
    for view in ["2ch", "4ch", "sa", "ras154"]:
        multi_model.zero_grad()
        x = torch.randn(2, 3, 219, 219)
        out = multi_model(x, dataset_type=view)
        loss = F.mse_loss(out, torch.randn_like(out))
        loss.backward()
        
        # Verify encoder has gradients
        encoder_zero = [n for n, p in multi_model.encoder.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
        assert len(encoder_zero) == 0, f"Encoder params without grad in view {view}: {encoder_zero}"
        
        # Verify active decoder has gradients
        active_dec_zero = [n for n, p in multi_model.decoders[view].named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
        assert len(active_dec_zero) == 0, f"Active decoder {view} params without grad: {active_dec_zero}"
        
        # Verify INACTIVE decoders have NO gradients (isolated computational graph)
        for other_view, other_dec in multi_model.decoders.items():
            if other_view != view:
                inactive_grad = [n for n, p in other_dec.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
                assert len(inactive_grad) == 0, f"Inactive decoder {other_view} unexpectedly received gradients: {inactive_grad}"
                
        print(f"  MultiDataset view '{view}': Shared encoder & target decoder gradients strictly isolated and active [PASS]")

    print("  [PASS] Full backward pass gradient flow verified.\n")


def test_attention_gate_bounds_and_stability():
    """Verify AttentionBlock gate weights alpha are strictly bounded in [0, 1] and numerically stable."""
    print("--- Test 5: Attention Gate Bounds alpha in [0, 1] & Numerical Stability ---")
    
    # We will hook into the Sigmoid of AttentionBlocks in ResUNetPlusPlusDecoder
    decoder = ResUNetPlusPlusDecoder(filters=[32, 64, 128, 256, 512], num_classes=5)
    decoder.eval()
    
    gate_records = []
    
    def make_hook(block_name):
        def hook_fn(module, input, output):
            # output of AttentionBlock.conv_attn is the gate alpha
            alpha = output
            gate_records.append({
                "name": block_name,
                "min": alpha.min().item(),
                "max": alpha.max().item(),
                "has_nan": torch.isnan(alpha).any().item(),
                "has_inf": torch.isinf(alpha).any().item(),
                "shape": list(alpha.shape)
            })
        return hook_fn
    
    hooks = []
    for name in ["attn1", "attn2", "attn3", "attn4"]:
        attn_module = getattr(decoder, name)
        h = attn_module.conv_attn.register_forward_hook(make_hook(name))
        hooks.append(h)
        
    test_inputs = [
        # Standard random normal
        ("Standard normal", torch.randn(2, 32, 219, 219)),
        # Extreme large values
        ("Large magnitude * 100", torch.randn(2, 32, 219, 219) * 100.0),
        # Extreme negative / positive
        ("Extreme constant 1e4", torch.ones(2, 32, 219, 219) * 10000.0),
        ("Extreme constant -1e4", torch.ones(2, 32, 219, 219) * -10000.0),
        # Zeros
        ("All zeros", torch.zeros(2, 32, 219, 219)),
    ]
    
    encoder = ResUNetPlusPlusEncoder(in_channels=32, filters=[32, 64, 128, 256, 512])
    encoder.eval()
    
    for desc, inp in test_inputs:
        gate_records.clear()
        x1, x2, x3, x4, x5 = encoder(inp)
        out = decoder(x1, x2, x3, x4, x5)
        
        assert len(gate_records) == 4, f"Expected 4 attention records, got {len(gate_records)}"
        for rec in gate_records:
            assert rec["min"] >= 0.0, f"Attention gate {rec['name']} min < 0: {rec['min']} in {desc}"
            assert rec["max"] <= 1.0, f"Attention gate {rec['name']} max > 1: {rec['max']} in {desc}"
            assert not rec["has_nan"], f"Attention gate {rec['name']} has NaN in {desc}"
            assert not rec["has_inf"], f"Attention gate {rec['name']} has Inf in {desc}"
            print(f"  Input: {desc:25s} | Gate: {rec['name']:5s} | Range: [{rec['min']:.6f}, {rec['max']:.6f}] | NaNs: 0, Infs: 0 [PASS]")

    for h in hooks:
        h.remove()
        
    print("  [PASS] Attention gate weights alpha in [0, 1] verified.\n")


def test_aspp_and_se_edge_cases():
    """Test ASPP and Squeeze-and-Excitation under edge cases like B=1, tiny spatial dims, etc."""
    print("--- Test 6: ASPP and SE Module Edge Cases ---")
    
    # 1. Batch size 1 on ASPP (verifying GroupNorm prevents BatchNorm failure on 1x1 image pool)
    aspp = ASPP(in_dims=256, out_dims=512, rate=[1, 2, 4])
    aspp.train()  # In training mode, BN with B=1 and 1x1 spatial would crash if used in image pool
    
    x_b1 = torch.randn(1, 256, 24, 24)
    out_b1 = aspp(x_b1)
    assert out_b1.shape == (1, 512, 24, 24), f"ASPP B=1 output shape mismatch: {out_b1.shape}"
    assert not torch.isnan(out_b1).any(), "NaN in ASPP output for B=1"
    print("  ASPP B=1 in training mode (GroupNorm safety): PASSED")
    
    # 2. SE block with channels not divisible by reduction ratio
    se = Squeeze_Excite_Block(channel=17, reduction=8)
    x_se = torch.randn(2, 17, 32, 32)
    out_se = se(x_se)
    assert out_se.shape == (2, 17, 32, 32)
    print("  SE block with prime channel count (17, reduction=8): PASSED")
    
    # 3. ASPP with odd/small spatial map
    x_odd = torch.randn(2, 256, 7, 7)
    out_odd = aspp(x_odd)
    assert out_odd.shape == (2, 512, 7, 7)
    print("  ASPP with small 7x7 feature map: PASSED")
    
    print("  [PASS] ASPP and SE edge cases verified.\n")


def run_all_stress_tests():
    print("=" * 70)
    print("STARTING CHALLENGER M2_1 (2D MODEL STRESS & RESOLUTION) AUDIT SUITE")
    print("=" * 70)
    
    test_encoder_five_stages()
    test_arbitrary_odd_resolutions_resunet2d()
    test_multidataset_resunet_plus_plus()
    test_full_backward_pass_all_parameters()
    test_attention_gate_bounds_and_stability()
    test_aspp_and_se_edge_cases()
    
    print("=" * 70)
    print("ALL 6 CHALLENGER STRESS TEST SUITES PASSED EMPIRICALLY WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_all_stress_tests()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
