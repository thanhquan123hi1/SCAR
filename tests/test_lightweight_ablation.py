"""Unit and checkpoint compatibility tests for Lightweight-M2-Plus Phase 1 ablations."""
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from training.dataset.data_contract import CLASS_NAMES
from training.models import (
    LightweightM2PlusNet,
    get_lightweight_testing,
    get_testing,
    model_from_config,
    validate_model_config,
)
from training.train import model_config_from_args, parse_args
from training.trainer.trainer import _configs_equal, _resume_model_config, load_checkpoint


ABLATION_CONFIGS = [
    ("lightweight_no_sspanet", 13_286_007),
    ("lightweight_no_skip_gates", 14_629_172),
    ("lightweight_no_pos_enc", 14_670_711),
    ("lightweight_no_film", 14_659_959),
    ("lightweight_no_cross_attention", 14_144_375),
    ("lightweight_resnet346", None),
]


def test_lightweight_ablation_yaml_configs_and_param_counts():
    base_args, base_merged = parse_args(["--config", "lightweight_m2_plus"])
    base_cfg = model_config_from_args(base_args, base_merged)
    base_model = model_from_config(base_cfg, img_size=128)
    base_params = sum(p.numel() for p in base_model.parameters())
    assert base_params == 14_670_711

    for cfg_name, expected_params in ABLATION_CONFIGS:
        args, merged = parse_args(["--config", cfg_name])
        cfg = model_config_from_args(args, merged)
        model = model_from_config(cfg, img_size=128)
        assert isinstance(model, LightweightM2PlusNet)
        n_params = sum(p.numel() for p in model.parameters())
        if expected_params is not None:
            assert n_params == expected_params, f"{cfg_name}: expected {expected_params}, got {n_params}"
        else:
            assert n_params < base_params

        if cfg_name == "lightweight_no_sspanet":
            assert isinstance(model.sspanet_cine, nn.Identity)
        elif cfg_name == "lightweight_no_skip_gates":
            assert model.gate_skip0 is None and model.gate_skip1 is None and model.gate_skip2 is None
        elif cfg_name == "lightweight_no_pos_enc":
            assert model.cross_attention.use_pos_enc is False
        elif cfg_name == "lightweight_no_film":
            assert model.encoder.film1 is None and model.encoder.film2 is None and model.encoder.film3 is None
        elif cfg_name == "lightweight_no_cross_attention":
            assert model.cross_attention.mha_scar is None and model.cross_attention.mha_edema is None
        elif cfg_name == "lightweight_resnet346":
            assert tuple(model.config.resnet.num_layers) == (3, 4, 6)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sspanet_gate": "identity"},
        {"use_skip_gates": False},
        {"use_pos_enc": False},
        {"use_film": False},
        {"use_cross_attention": False},
        {"resnet_num_layers": (1, 1, 1)},
    ],
)
def test_lightweight_ablation_forward_backward_and_checkpoint_roundtrip(tmp_path, overrides):
    cfg = get_lightweight_testing()
    for k, v in overrides.items():
        if k == "resnet_num_layers":
            cfg.resnet.num_layers = v
        else:
            setattr(cfg, k, v)
    source = model_from_config(cfg, img_size=32)
    imgs = [torch.randn(2, 1, 32, 32) for _ in range(3)]
    out = source(*imgs)
    assert out.shape == (2, 4, 32, 32)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in source.parameters() if p.requires_grad)

    source.eval()
    ckpt_path = tmp_path / "ablation.pt"
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


def test_legacy_lightweight_resume_metadata_compat_and_mismatch():
    default_cfg = get_lightweight_testing().to_dict()
    default_cfg["architecture"] = "lightweight_m2_plus"
    legacy_cfg = dict(default_cfg)
    for k in ("use_skip_gates", "use_film", "use_cross_attention"):
        legacy_cfg.pop(k)
    assert _configs_equal(_resume_model_config(default_cfg), _resume_model_config(legacy_cfg))

    for flag in ("use_skip_gates", "use_film", "use_cross_attention"):
        modified = dict(default_cfg) | {flag: False}
        assert not _configs_equal(_resume_model_config(default_cfg), _resume_model_config(modified))


def test_invalid_ablation_flag_values_rejected():
    for flag in ("use_skip_gates", "use_film", "use_cross_attention"):
        bad = get_lightweight_testing().to_dict() | {flag: "false"}
        with pytest.raises(ValueError, match=flag):
            validate_model_config(bad)

        bad_m2 = get_testing().to_dict() | {flag: False}
        with pytest.raises(ValueError, match=flag):
            validate_model_config(bad_m2)

    bad_gate = get_lightweight_testing()
    bad_gate.sspanet_gate = "centered_tanh_v2"
    with pytest.raises(ValueError, match="optimized or identity"):
        LightweightM2PlusNet(bad_gate, img_size=32)


def test_pretrained_loads_for_truncated_and_no_film_lightweight():
    weights_path = Path("/content/SCAR/training/pretrained/R50-ViT-B_16.npz")
    if not weights_path.is_file():
        pytest.skip("Pretrained weights not present")
    for cfg_name in ("lightweight_no_film", "lightweight_resnet346"):
        args, merged = parse_args(["--config", cfg_name])
        cfg = model_config_from_args(args, merged)
        model = model_from_config(cfg, img_size=128)
        model.load_pretrained_encoders(weights_path)
