"""Equation semantics, numerical boundaries and checkpoint integration."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from training.dataset.data_contract import CLASS_NAMES
from training.loss.losses import SegmentationLoss
from training.models import M2PlusNet, get_testing, model_from_config
from training.models.modules.sspanet import PAPER_FORMULA_GATE, SSPANet_Block
from training.models.modules.sspanet_formula import PaperFormulaSSPANet
from training.train import model_config_from_args, parse_args
from training.trainer.trainer import _configs_equal, _resume_model_config, load_checkpoint


def test_sequential_equations_match_explicit_broadcast_reference_and_gradients():
    torch.manual_seed(19)
    block = PaperFormulaSSPANet(4).double().eval()
    x = torch.randn(2, 4, 3, 5, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    output, mc, ms = block(x, return_attention=True)
    # Deliberately materialize concatenated 2D strips, independently of strip_logits.
    descriptor = torch.cat((reference_x.mean((2, 3), keepdim=True),
                            reference_x.amax((2, 3), keepdim=True)), dim=1)
    channel = torch.sigmoid(block.channel_norm(block.channel_proj(descriptor)))
    refined = reference_x * channel
    sh = refined.mean(2, keepdim=True).expand_as(refined)
    sv = refined.mean(3, keepdim=True).expand_as(refined)
    strip = block.strip_proj(torch.cat((sh, sv), dim=1))
    mean = refined.mean(1, keepdim=True)
    std = (refined.var(1, correction=0, keepdim=True) + 1e-6).sqrt()
    style = block.style_proj(torch.cat((mean, std), dim=1))
    spatial = torch.sigmoid(block.spatial_proj(strip + style))
    expected = refined * spatial
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    assert mc.shape == (2, 4, 1, 1) and ms.shape == (2, 1, 3, 5)
    assert torch.all(output.abs() <= x.abs())  # equation (17), no residual gain
    output.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-11, atol=1e-12)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())


def test_style_uses_population_statistics_across_channels_at_each_pixel():
    spatial = torch.tensor([[0., 10.], [100., 1000.]])
    x = torch.stack([spatial + offset for offset in (1., 2., 3.)])[None]
    stats = PaperFormulaSSPANet(3).style_statistics(x)
    torch.testing.assert_close(stats[:, 0], (spatial + 2)[None])
    torch.testing.assert_close(stats[:, 1], torch.full((1, 2, 2), (2 / 3 + 1e-6) ** 0.5))


def test_strip_projection_before_broadcast_preserves_bias_and_gradients():
    torch.manual_seed(3)
    block = PaperFormulaSSPANet(7).double()
    x = torch.randn(2, 7, 3, 9, dtype=torch.float64, requires_grad=True)
    block.strip_proj.bias.data.fill_(2.5)
    explicit = block.strip_proj(torch.cat((x.mean(2, keepdim=True).expand_as(x),
                                          x.mean(3, keepdim=True).expand_as(x)), 1))
    projected = block.strip_logits(x)
    torch.testing.assert_close(projected, explicit, rtol=1e-12, atol=1e-12)
    first = torch.autograd.grad(projected.square().sum(), (x, block.strip_proj.weight, block.strip_proj.bias))
    second = torch.autograd.grad(explicit.square().sum(), (x, block.strip_proj.weight, block.strip_proj.bias))
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (1, 4, 1, 7), (2, 4, 3, 5)])
@pytest.mark.parametrize("amp", [False, True])
def test_constant_features_singleton_batches_and_amp_remain_finite(shape, amp):
    block = PaperFormulaSSPANet(shape[1]).train()
    x = torch.full(shape, 300., requires_grad=True)
    tracked = block.channel_norm.num_batches_tracked.clone()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        y = block(x)
        loss = y.float().square().mean()
    assert y.shape == x.shape and torch.isfinite(y).all()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    assert int(block.channel_norm.num_batches_tracked - tracked) == (shape[0] > 1)


@pytest.mark.parametrize("dataset", ["myops380", "myopspp_unet2d_roi128_1mm"])
@pytest.mark.parametrize("name,gate", [("m2_plus_paper_formula", PAPER_FORMULA_GATE),
                                      ("m2_plus_legacy_sspanet", "legacy_v1"),
                                      ("m2_plus_no_sspanet", "identity")])
def test_experiment_configs_use_shared_training_resolution(dataset, name, gate):
    args, merged = parse_args(["--config", name, "--dataset", dataset])
    config = model_config_from_args(args, merged)
    assert config.sspanet_gate == gate
    assert config.resnet.num_layers == (3, 4, 9)
    assert tuple(config.decoder_channels) == (256, 128, 64, 16)
    assert args.max_epochs == 300 and args.batch_size == 16 and args.accum_steps == 1
    assert merged["model"]["sspanet_gate"] == gate


def test_full_model_backpropagates_through_all_formula_branches_and_encoders():
    torch.set_num_threads(2)
    config = get_testing()
    config.sspanet_gate = PAPER_FORMULA_GATE
    model = M2PlusNet(config, img_size=32)
    images = [torch.rand(2, 1, 32, 48) for _ in range(3)]
    output = model(*images)
    assert output.shape == (2, 4, 32, 48)
    loss = SegmentationLoss()(output, torch.randint(0, 4, (2, 32, 48)))["loss"]
    loss.backward()
    for block in (model.sspanet_cine, model.sspanet_psir, model.sspanet_t2w):
        assert isinstance(block.paper_formula, PaperFormulaSSPANet)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
        assert all(p.grad.abs().sum() > 0 for p in block.parameters())
    for encoder in (model.encoder_cine, model.encoder_psir, model.encoder_t2w):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())


@pytest.mark.parametrize("gate", [PAPER_FORMULA_GATE, "identity"])
def test_checkpoint_roundtrip_and_resume_identity_preserve_variant(tmp_path, gate):
    config = get_testing()
    config.sspanet_gate = gate
    source = model_from_config(config, img_size=32).eval()
    metadata = source.config.to_dict()
    path = tmp_path / "model.pt"
    torch.save(dict(format_version=1, class_names=CLASS_NAMES, model_config=metadata,
                    model=source.state_dict()), path)
    saved = load_checkpoint(path)
    restored = model_from_config(saved["model_config"], img_size=32).eval()
    restored.load_state_dict(saved["model"], strict=True)
    images = [torch.rand(1, 1, 32, 32) for _ in range(3)]
    with torch.no_grad():
        torch.testing.assert_close(source(*images), restored(*images), rtol=0, atol=0)
    changed = deepcopy(metadata) | {"sspanet_gate": "centered_tanh_v2"}
    assert not _configs_equal(_resume_model_config(metadata), _resume_model_config(changed))
    if gate == "identity":
        assert sum(p.numel() for p in source.sspanet_cine.parameters()) == 0


@pytest.mark.parametrize("saved_gate,weights_gate", [(None, PAPER_FORMULA_GATE),
                                                   ("legacy_v1", PAPER_FORMULA_GATE),
                                                   (PAPER_FORMULA_GATE, "centered_tanh_v2")])
def test_loader_rejects_missing_or_conflicting_paper_metadata(tmp_path, saved_gate, weights_gate):
    config = get_testing()
    config.sspanet_gate = weights_gate
    source = M2PlusNet(config, img_size=32)
    metadata = source.config.to_dict()
    if saved_gate is None:
        metadata.pop("sspanet_gate")
    else:
        metadata["sspanet_gate"] = saved_gate
    path = tmp_path / "bad.pt"
    torch.save(dict(format_version=1, class_names=CLASS_NAMES, model_config=metadata,
                    model=source.state_dict()), path)
    with pytest.raises(ValueError, match="sspanet_gate"):
        load_checkpoint(path)


def test_formula_state_cannot_be_loaded_into_previous_structure():
    paper = SSPANet_Block(8, gate_mode=PAPER_FORMULA_GATE)
    old = SSPANet_Block(8, gate_mode="legacy_v1")
    with pytest.raises(RuntimeError, match="Missing key"):
        old.load_state_dict(paper.state_dict(), strict=True)
