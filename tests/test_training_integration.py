"""M2-Plus configuration, loss, checkpoint and training regression checks."""
from argparse import Namespace

import pytest
import torch
import yaml

from training.dataset.data_contract import CLASS_NAMES
from training.loss.losses import SegmentationLoss
from training.models import M2PlusNet, build_model, get_testing, model_from_config
from training.train import _load_initial_weights, parse_args


@pytest.mark.parametrize("dataset", ["myops380", "myopspp_bc80"])
def test_loss_yaml_and_cli_use_the_same_settings(tmp_path, dataset):
    loss = dict(scar_weight=0.2, aar_weight=0.3,
                dice_class_weights=[0, 1, 2, 2], ce_class_weights=[0.2, 1, 2, 2])
    config = tmp_path / "model.yaml"
    config.write_text(yaml.safe_dump({"model": {"ablation": "M2-PLUS"}, "loss": loss,
                                     "train": {"init_weights": None}}))
    args, _ = parse_args(["--config", str(config), "--dataset", dataset])
    for key, value in loss.items():
        assert getattr(args, key) == value
    explicit, _ = parse_args(["--config", str(config), "--dataset", dataset,
                             "--scar-weight", "0.4", "--ce-class-weights", "1", "1", "1", "1"])
    assert explicit.scar_weight == 0.4
    assert explicit.ce_class_weights == [1, 1, 1, 1]


@pytest.mark.parametrize("flag,value", [
    ("--aar-weight", "-1"), ("--aar-weight", "nan"),
    ("--scar-weight", "-1"), ("--scar-weight", "inf"),
    ("--ce-class-weights", "1 2 3"), ("--ce-class-weights", "0 0 0 0"),
    ("--ce-class-weights", "1 1 -1 1"), ("--dice-class-weights", "1 1 nan 1"),
])
def test_invalid_loss_settings_are_rejected_before_training(flag, value):
    with pytest.raises(SystemExit):
        parse_args(["--config", "testing", flag, *value.split()])


@pytest.mark.parametrize("other", ["--resume", "--pretrained"])
def test_initialization_sources_cannot_be_combined(other):
    with pytest.raises(SystemExit):
        parse_args(["--config", "testing", "--init-weights", "init.pt", other, "other.pt"])


def test_zero_head_initialization():
    model = M2PlusNet(get_testing(), img_size=32, zero_head=True)
    assert torch.count_nonzero(model.segmentation_head[0].weight) == 0
    assert torch.count_nonzero(model.segmentation_head[0].bias) == 0


def test_resume_requires_the_same_loss_and_supports_old_default_checkpoints():
    from training.trainer.trainer import _validate_resume_loss_config
    defaults = dict(aar_weight=0.0, scar_weight=0.0,
                    dice_class_weights=None, ce_class_weights=None)
    _validate_resume_loss_config({}, Namespace(**defaults))
    for key in defaults:
        changed = defaults | {key: [1, 1, 2, 2] if "class" in key else 0.25}
        with pytest.raises(ValueError, match=key):
            _validate_resume_loss_config({}, Namespace(**changed))
        with pytest.raises(ValueError, match=key):
            _validate_resume_loss_config(changed, Namespace(**defaults))


def test_corrected_loss_objective_cannot_silently_resume_old_weighted_runs():
    from training.trainer.trainer import _validate_resume_loss_config
    args = Namespace(ce_class_weights=[0.2, 1, 2, 2])
    with pytest.raises(ValueError, match="loss protocol"):
        _validate_resume_loss_config(vars(args), args, saved_version=1)
    _validate_resume_loss_config(vars(args), args, saved_version=2)
    _validate_resume_loss_config({}, Namespace(), saved_version=1)


def test_forward_backward_and_eval_contract():
    torch.set_num_threads(2)
    model = M2PlusNet(get_testing(), img_size=32)
    images = [torch.randn(2, 1, 32, 32) for _ in range(3)]
    target = torch.randint(0, 4, (2, 32, 32))
    output = model(*images)
    loss = SegmentationLoss(aar_weight=0.2, scar_weight=0.2)(output, target)
    assert torch.isfinite(loss["loss"])
    loss["loss"].backward()
    for encoder in (model.encoder_cine, model.encoder_psir, model.encoder_t2w):
        assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in encoder.parameters())
    model.eval()
    with torch.no_grad():
        prediction = model(*images)
    assert prediction.shape == (2, 4, 32, 32)
    assert torch.isfinite(prediction).all()


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


def test_pathology_loss_uses_float32_reductions_under_half_precision():
    target = torch.full((1, 256, 256), 2, dtype=torch.long)
    logits = torch.zeros(1, 4, 256, 256, dtype=torch.float16, requires_grad=True)
    criterion = SegmentationLoss(aar_weight=0.2, scar_weight=0.2)
    losses = criterion(logits, target)
    expected = criterion(logits.detach().float(), target)
    torch.testing.assert_close(losses["loss"], expected["loss"])
    losses["loss"].backward()
    assert torch.isfinite(logits.grad).all()


def test_validation_epoch_does_not_update_batchnorm():
    from torch.utils.data import DataLoader
    from training.trainer.trainer import run_epoch
    model = M2PlusNet(get_testing(), img_size=32)
    samples = [{"image": torch.randn(1, 32, 32), "image1": torch.randn(1, 32, 32),
                "image2": torch.randn(1, 32, 32), "label": torch.randint(0, 4, (32, 32))} for _ in range(2)]
    bn = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = bn.running_mean.clone()
    metrics, _ = run_epoch(model, DataLoader(samples, batch_size=2),
                           SegmentationLoss(aar_weight=0.2), torch.device("cpu"), None)
    assert not model.training
    torch.testing.assert_close(bn.running_mean, before)
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_initial_weights_require_canonical_labels_and_exact_m2_plus_parameters(tmp_path):
    source = M2PlusNet(get_testing(), img_size=32)
    destination = M2PlusNet(get_testing(), img_size=32)
    path = tmp_path / "source.pt"
    checkpoint = dict(format_version=1, class_names=CLASS_NAMES,
                      model=source.state_dict(), model_config=source.config.to_dict())
    torch.save(checkpoint, path)
    assert _load_initial_weights(destination, path) == ([], [])
    for name, value in source.state_dict().items():
        torch.testing.assert_close(destination.state_dict()[name], value)
    checkpoint["class_names"] = ("background", "normal_myocardium", "scar", "edema")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="label semantics"):
        _load_initial_weights(destination, path)
    checkpoint["class_names"] = CLASS_NAMES
    checkpoint["model"].pop("cross_fusion.mha_scar.in_proj_weight")
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        _load_initial_weights(destination, path)


@pytest.mark.parametrize("name", ["m2_plus", "m2plus", "M2-Plus"])
def test_m2_plus_names_select_the_same_architecture(name):
    assert isinstance(build_model(name, config=get_testing(), img_size=32), M2PlusNet)


@pytest.mark.parametrize("ablation", ["M2", "M3", "M2-MAX", "M2-MAX-V7", "H-CMSPA"])
def test_retired_models_are_rejected_by_cli_and_checkpoint_factory(ablation):
    with pytest.raises(SystemExit):
        parse_args(["--config", "testing", "--ablation", ablation])
    config = get_testing()
    config.ablation = ablation
    with pytest.raises(ValueError, match="Only M2-Plus"):
        model_from_config(config, img_size=32)


def test_legacy_m2_plus_checkpoint_metadata_remains_supported():
    from training.trainer.trainer import _configs_equal, _resume_model_config
    config = get_testing()
    legacy = config.to_dict() | {"architecture": "cmspa_net", "ablation": "M2-Plus"}
    assert isinstance(model_from_config(legacy, img_size=32), M2PlusNet)
    assert _configs_equal(_resume_model_config(legacy), _resume_model_config(config.to_dict()))
    with pytest.raises(ValueError, match="must identify"):
        model_from_config({"architecture": "cmspa_net"})
    with pytest.raises(ValueError, match="Only M2-Plus"):
        build_model("cross_attn_baseline", config=config)


def test_default_configuration_is_m2_plus():
    args, merged = parse_args([])
    assert args.ablation == "M2-PLUS"
    assert merged["model"]["model_name"] == "M2-Plus"


@pytest.mark.parametrize("key", ["wall_weight", "inclusion_weight"])
def test_removed_head_losses_are_rejected_in_yaml(tmp_path, key):
    config = tmp_path / "removed.yaml"
    config.write_text(yaml.safe_dump({"loss": {key: 0.2}}))
    with pytest.raises(ValueError, match="Unknown configuration key"):
        parse_args(["--config", str(config)])


def test_checkpoint_loader_rejects_retired_and_unidentified_models(tmp_path):
    from training.trainer.trainer import load_checkpoint
    checkpoint = tmp_path / "unsupported.pt"
    payload = dict(format_version=1, class_names=CLASS_NAMES, model={})
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="identifying M2-PLUS"):
        load_checkpoint(checkpoint)
    payload["model_config"] = {"ablation": "M3"}
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="Only M2-Plus"):
        load_checkpoint(checkpoint)


def test_pipeline_rejects_retired_models_before_preprocessing(tmp_path):
    from unittest.mock import patch
    from run_all import main
    config = tmp_path / "retired.yaml"
    config.write_text(yaml.safe_dump({"model": {"ablation": "M3"}}))
    with patch("run_all.run_command") as command:
        with pytest.raises(ValueError, match="Only M2-Plus"):
            main(["--config", str(config)])
    command.assert_not_called()
