"""Unit, gradient, benchmark, and checkpoint compatibility tests for Phase 2 Lightweight M2-Plus architectures."""
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from training.benchmark import counted_flops
from training.dataset.data_contract import CLASS_NAMES
from training.models import (
    LightweightM2PlusNet,
    get_lightweight_testing,
    get_testing,
    model_from_config,
    validate_model_config,
)
from training.models.modules.decoder import BoundaryRefinementBlock
from training.models.modules.fusion import TriModalSkipFusion
from training.models.modules.m2_plus import LocalModalityContrastFusion
from training.train import model_config_from_args, parse_args
from training.trainer.trainer import _configs_equal, _resume_model_config, load_checkpoint


PHASE2_CONFIGS = [
    ("lightweight_p2_clean_pruned", 12_759_671),
    ("lightweight_p2_contrast_fusion", None),
    ("lightweight_p2_adaptive_skip", None),
    ("lightweight_p2_v2", None),
]


def test_phase2_yaml_configs_and_param_budgets():
    base_args, base_merged = parse_args(["--config", "lightweight_m2_plus"])
    base_model = model_from_config(model_config_from_args(base_args, base_merged), img_size=128)
    base_params = sum(p.numel() for p in base_model.parameters())

    counts = {}
    for cfg_name, expected in PHASE2_CONFIGS:
        args, merged = parse_args(["--config", cfg_name])
        cfg = model_config_from_args(args, merged)
        model = model_from_config(cfg, img_size=128)
        assert isinstance(model, LightweightM2PlusNet)
        n_params = sum(p.numel() for p in model.parameters())
        counts[cfg_name] = n_params
        if expected is not None:
            assert n_params == expected, f"{cfg_name}: expected {expected}, got {n_params}"
        assert n_params < base_params, f"{cfg_name} ({n_params:,}) must stay below Baseline ({base_params:,})"

        if cfg_name == "lightweight_p2_clean_pruned":
            assert isinstance(model.sspanet_cine, nn.Identity)
            assert model.cross_attention.mha_scar is None
            assert model.skip_fusions is None
            assert model.boundary_refine is None
        elif cfg_name == "lightweight_p2_contrast_fusion":
            assert isinstance(model.cross_attention, LocalModalityContrastFusion)
            assert model.skip_fusions is None
            assert model.boundary_refine is None
        elif cfg_name == "lightweight_p2_adaptive_skip":
            assert model.skip_fusions is not None
            assert all(isinstance(m, TriModalSkipFusion) for m in model.skip_fusions)
            assert model.boundary_refine is None
        elif cfg_name == "lightweight_p2_v2":
            assert isinstance(model.cross_attention, LocalModalityContrastFusion)
            assert model.skip_fusions is not None
            assert isinstance(model.boundary_refine, BoundaryRefinementBlock)

    assert counts["lightweight_p2_adaptive_skip"] > counts["lightweight_p2_clean_pruned"]
    assert counts["lightweight_p2_v2"] > counts["lightweight_p2_contrast_fusion"]


def test_trimodal_skip_fusion_uniform_init_and_multistep_grad():
    torch.manual_seed(0)
    mod = TriModalSkipFusion(channels=32, reduction=8)
    c, p, t = [torch.randn(2, 32, 16, 16) for _ in range(3)]
    out_init = mod(c, p, t)
    torch.testing.assert_close(out_init, (c + p + t) / 3.0, rtol=1e-5, atol=1e-6)

    opt = torch.optim.SGD(mod.parameters(), lr=0.1)
    target = p * 2.0 - t
    for _ in range(2):
        opt.zero_grad()
        loss = (mod(c, p, t) - target).square().mean()
        loss.backward()
        opt.step()
    for name, param in mod.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.abs().sum() > 0, f"{name} did not receive updates after 2 steps"


@pytest.mark.parametrize(
    "overrides",
    [
        {"sspanet_gate": "identity", "use_cross_attention": False, "use_pos_enc": False},
        {"bottleneck_fusion": "local_contrast", "latent_dim": 48},
        {"skip_fusion": "modality_weighted"},
        {"decoder_refinement": "boundary"},
        {
            "sspanet_gate": "identity",
            "bottleneck_fusion": "local_contrast",
            "latent_dim": 48,
            "skip_fusion": "modality_weighted",
            "decoder_refinement": "boundary",
        },
        {
            "sspanet_gate": "optimized",
            "bottleneck_fusion": "local_contrast",
            "latent_dim": 48,
            "skip_fusion": "modality_weighted",
            "decoder_refinement": "boundary",
        },
    ],
)
def test_phase2_forward_backward_flops_and_checkpoint_roundtrip(tmp_path, overrides):
    cfg = get_lightweight_testing()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    source = model_from_config(cfg, img_size=32)
    imgs = [torch.randn(2, 1, 32, 32) for _ in range(3)]
    out = source(*imgs)
    assert out.shape == (2, 4, 32, 32)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in source.parameters() if p.requires_grad)

    flops_info = counted_flops(source, tuple(x[:1] for x in imgs))
    assert flops_info["macs"] > 0

    source.eval()
    ckpt_path = tmp_path / "phase2.pt"
    torch.save(
        {
            "format_version": 1,
            "class_names": CLASS_NAMES,
            "model_config": source.config.to_dict(),
            "model": source.state_dict(),
        },
        ckpt_path,
    )
    loaded = load_checkpoint(ckpt_path)
    restored = model_from_config(loaded["model_config"], img_size=32).eval()
    restored.load_state_dict(loaded["model"], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(source(*imgs), restored(*imgs), rtol=0, atol=0)


def test_phase2_resume_metadata_compat_and_validation_errors():
    default_cfg = get_lightweight_testing().to_dict()
    default_cfg["architecture"] = "lightweight_m2_plus"
    legacy_cfg = dict(default_cfg)
    for key in ("bottleneck_fusion", "skip_fusion", "decoder_refinement"):
        legacy_cfg.pop(key)
    assert _configs_equal(_resume_model_config(default_cfg), _resume_model_config(legacy_cfg))

    for key, val in (
        ("bottleneck_fusion", "local_contrast"),
        ("skip_fusion", "modality_weighted"),
        ("decoder_refinement", "boundary"),
    ):
        changed = dict(default_cfg) | {key: val}
        assert not _configs_equal(_resume_model_config(default_cfg), _resume_model_config(changed))

        bad_val = dict(default_cfg) | {key: "unknown_mode"}
        with pytest.raises(ValueError, match=key):
            validate_model_config(bad_val)

        bad_arch = get_testing().to_dict() | {key: val}
        with pytest.raises(ValueError, match=key):
            validate_model_config(bad_arch)


def test_phase2_pretrained_loads_and_exact_param_prints():
    weights_path = Path("/content/SCAR/training/pretrained/R50-ViT-B_16.npz")
    if not weights_path.is_file():
        pytest.skip("Pretrained weights not present")
    for cfg_name, _ in PHASE2_CONFIGS:
        args, merged = parse_args(["--config", cfg_name])
        cfg = model_config_from_args(args, merged)
        model = model_from_config(cfg, img_size=128)
        model.load_pretrained_encoders(weights_path)
