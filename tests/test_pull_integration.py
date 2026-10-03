"""Regression checks for new M2-Max models with the existing dataset pipeline."""
from argparse import Namespace

import pytest
import torch
import yaml

from training.train import parse_args
from training.models.cmspa_net import CMSPANet, get_testing
from training.loss.losses import SegmentationLoss


@pytest.mark.parametrize("dataset", ["myops380", "myopspp_bc80"])
def test_new_loss_yaml_and_cli_use_the_same_settings(tmp_path, dataset):
    loss = dict(scar_weight=0.2, wall_weight=0.3, inclusion_weight=0.1,
                dice_class_weights=[0, 1, 2, 2], ce_class_weights=[0.2, 1, 2, 2])
    config = tmp_path / "model.yaml"
    config.write_text(yaml.safe_dump({"model": {"ablation": "M2-MAX-V7"}, "loss": loss,
                                     "train": {"init_weights": None}}))
    args, _ = parse_args(["--config", str(config), "--dataset", dataset])
    for key, value in loss.items():
        assert getattr(args, key) == value
    explicit, _ = parse_args(["--config", str(config), "--dataset", dataset,
                             "--scar-weight", "0.4", "--ce-class-weights", "1", "1", "1", "1"])
    assert explicit.scar_weight == 0.4
    assert explicit.ce_class_weights == [1, 1, 1, 1]


@pytest.mark.parametrize("flag,value", [
    ("--wall-weight", "-1"), ("--wall-weight", "nan"),
    ("--inclusion-weight", "-1"), ("--inclusion-weight", "inf"),
    ("--ce-class-weights", "1 2 3"), ("--ce-class-weights", "0 0 0 0"),
    ("--ce-class-weights", "1 1 -1 1"), ("--dice-class-weights", "1 1 nan 1"),
])
def test_invalid_loss_settings_are_rejected_before_training(flag, value):
    with pytest.raises(SystemExit):
        parse_args(["--config", "testing", flag, *value.split()])


@pytest.mark.parametrize("other", ["--resume", "--pretrained"])
def test_initialization_sources_cannot_be_combined(other):
    with pytest.raises(SystemExit):
        parse_args(["--config", "testing", "--init-weights", "init.pth", other, "other.pth"])


@pytest.mark.parametrize("ablation", ["M2-MAX-V6", "M2-MAX-V7"])
def test_hierarchical_zero_head_initialization(ablation):
    model = CMSPANet(get_testing(), img_size=32, ablation=ablation, zero_head=True)
    assert torch.count_nonzero(model.segmentation_head.base_head.weight) == 0
    assert torch.count_nonzero(model.segmentation_head.base_head.bias) == 0


def test_resume_requires_the_same_loss_and_supports_old_default_checkpoints():
    from training.trainer.trainer import _validate_resume_loss_config
    defaults = dict(aar_weight=0.0, scar_weight=0.0, wall_weight=0.0,
                    inclusion_weight=0.0, dice_class_weights=None, ce_class_weights=None)
    _validate_resume_loss_config({}, Namespace(**defaults))
    for key in defaults:
        changed = defaults | {key: [1, 1, 2, 2] if "class" in key else 0.25}
        with pytest.raises(ValueError, match=key):
            _validate_resume_loss_config({}, Namespace(**changed))
        with pytest.raises(ValueError, match=key):
            _validate_resume_loss_config(changed, Namespace(**defaults))


def test_corrected_loss_objective_cannot_silently_resume_old_weighted_runs():
    from training.trainer.trainer import _validate_resume_loss_config
    args = Namespace(inclusion_weight=0.1)
    with pytest.raises(ValueError, match="loss protocol"):
        _validate_resume_loss_config(vars(args), args, saved_version=1)
    _validate_resume_loss_config(vars(args), args, saved_version=2)
    _validate_resume_loss_config({}, Namespace(), saved_version=1)


@pytest.mark.parametrize("ablation", ["M2-MAX-PRO", "M2-MAX-V2", "M2-MAX-V4",
                                      "M2-MAX-V5", "M2-MAX-V6", "M2-MAX-V7"])
def test_new_models_forward_backward_and_eval_contract(ablation):
    torch.set_num_threads(2)
    model = CMSPANet(get_testing(), img_size=32, ablation=ablation)
    images = [torch.randn(2, 1, 32, 32) for _ in range(3)]
    target = torch.randint(0, 4, (2, 32, 32))
    output = model(*images)
    loss = SegmentationLoss(aar_weight=0.2, scar_weight=0.2,
                            wall_weight=0.3 if ablation in ("M2-MAX-V6", "M2-MAX-V7") else 0.0)(output, target)
    assert torch.isfinite(loss["loss"])
    loss["loss"].backward()
    for encoder in (model.encoder_cine, model.encoder_psir, model.encoder_t2w):
        assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in encoder.parameters())
    model.eval()
    with torch.no_grad():
        prediction = model(*images)
    assert prediction.shape == (2, 4, 32, 32)
    assert torch.isfinite(prediction).all()


def test_inclusion_penalizes_disagreement_with_independent_anatomy_streams():
    logits = torch.zeros(1, 4, 2, 2, requires_grad=True)
    wall = torch.full((1, 1, 2, 2), -4.0, requires_grad=True)
    aar = torch.full((1, 1, 2, 2), 4.0, requires_grad=True)
    criterion = SegmentationLoss(inclusion_weight=0.5)
    loss = criterion({"logits": logits, "wall_logits": wall, "aar_logits": aar},
                     torch.ones(1, 2, 2, dtype=torch.long))["inclusion_loss"]
    assert loss.item() > 0.4
    loss.backward()
    assert wall.grad.abs().sum() > 0
    assert aar.grad.abs().sum() > 0
    satisfied = criterion({"logits": logits.detach(), "wall_logits": -wall.detach(),
                           "aar_logits": torch.zeros_like(aar)}, torch.zeros(1, 2, 2, dtype=torch.long))
    assert satisfied["inclusion_loss"].item() == 0.0


@pytest.mark.parametrize("weight", ["wall_weight", "inclusion_weight"])
def test_auxiliary_anatomy_loss_requires_an_independent_head(weight):
    with pytest.raises(ValueError, match="wall_logits"):
        SegmentationLoss(**{weight: 0.2})(torch.zeros(1, 4, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long))


def test_weighted_ce_is_finite_on_a_slice_with_only_zero_weight_classes():
    logits = torch.randn(2, 4, 4, 4, requires_grad=True)
    target = torch.zeros(2, 4, 4, dtype=torch.long)
    losses = SegmentationLoss(ce_class_weights=[0, 1, 2, 2])(logits, target)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert torch.isfinite(logits.grad).all()


def test_weighted_ce_reduction_matches_sample_weighted_accumulation():
    logits = torch.randn(3, 4, 4, 4)
    target = torch.stack([torch.full((4, 4), i, dtype=torch.long) for i in (0, 1, 3)])
    criterion = SegmentationLoss(ce_class_weights=[0.2, 1, 2, 4])
    full = criterion(logits, target)["ce"]
    accumulated = sum(criterion(logits[i:i+1], target[i:i+1])["ce"] for i in range(3)) / 3
    torch.testing.assert_close(full, accumulated)


def test_auxiliary_loss_uses_float32_reductions_under_half_precision():
    target = torch.full((1, 256, 256), 2, dtype=torch.long)
    output = {"logits": torch.zeros(1, 4, 256, 256, dtype=torch.float16),
              "wall_logits": torch.full((1, 1, 256, 256), 12, dtype=torch.float16),
              "aar_logits": torch.full((1, 1, 256, 256), 12, dtype=torch.float16)}
    criterion = SegmentationLoss(aar_weight=0.2, wall_weight=0.3)
    losses = criterion(output, target)
    assert losses["wall_loss"].item() < 0.001
    expected = criterion({key: value.float() for key, value in output.items()}, target)
    torch.testing.assert_close(losses["loss"], expected["loss"])


@pytest.mark.parametrize("ablation", ["M2-MAX-V6", "M2-MAX-V7"])
def test_validation_epoch_keeps_auxiliary_outputs_without_switching_batchnorm(ablation):
    from torch.utils.data import DataLoader
    from training.trainer.trainer import run_epoch
    model = CMSPANet(get_testing(), img_size=32, ablation=ablation)
    samples = [{"image": torch.randn(1, 32, 32), "image1": torch.randn(1, 32, 32),
                "image2": torch.randn(1, 32, 32), "label": torch.randint(0, 4, (32, 32))} for _ in range(2)]
    bn = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = bn.running_mean.clone()
    metrics, _ = run_epoch(model, DataLoader(samples, batch_size=2),
                           SegmentationLoss(wall_weight=0.2, aar_weight=0.2, inclusion_weight=0.1),
                           torch.device("cpu"), None)
    assert not model.training
    torch.testing.assert_close(bn.running_mean, before)
    assert metrics["wall_loss"] > 0
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_initial_weights_transfer_canonical_head_to_hierarchical_model(tmp_path):
    from training.train import _load_initial_weights
    from training.dataset.data_contract import CLASS_NAMES
    source = CMSPANet(get_testing(), img_size=32, ablation="M2-MAX-V4")
    destination = CMSPANet(get_testing(), img_size=32, ablation="M2-MAX-V6")
    path = tmp_path / "source.pth"
    torch.save({"format_version": 1, "class_names": CLASS_NAMES, "model": source.state_dict()}, path)
    _load_initial_weights(destination, path)
    torch.testing.assert_close(source.segmentation_head[0].weight, destination.segmentation_head.base_head.weight)
    torch.testing.assert_close(next(source.encoder_cine.parameters()), next(destination.encoder_cine.parameters()))
    # Both native dataset profiles use CLASS_NAMES after loader canonicalization.
    torch.save({"format_version": 1, "class_names": ("background", "normal_myocardium", "scar", "edema"),
                "model": source.state_dict()}, path)
    before = destination.segmentation_head.base_head.weight.detach().clone()
    with pytest.raises(ValueError, match="label semantics"):
        _load_initial_weights(destination, path)
    torch.testing.assert_close(destination.segmentation_head.base_head.weight, before)


@pytest.mark.parametrize("name,ablation", [("m2_max_v2", "M2-MAX-V2"), ("m2_max_v7", "M2-MAX-V7")])
def test_new_models_are_available_through_registry(name, ablation):
    from training.models import build_model
    model = build_model(name, config=get_testing(), img_size=32)
    assert model.ablation == ablation
