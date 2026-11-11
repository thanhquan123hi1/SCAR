"""Independent reference checks for SCAR's I-MMSeg training recipe."""
from copy import deepcopy
from pathlib import Path
import runpy

import pytest
import torch

from training.loss.losses import DiceLoss, SegmentationLoss, LOSS_PROTOCOL_VERSION
from training.optimization import (
    OPTIMIZATION_PROTOCOL, build_optimizer, build_scheduler, validate_optimization_checkpoint,
)
from training.train import parse_args
from training.trainer.trainer import _validate_resume_loss_config


REFERENCE_DICE = runpy.run_path(str(Path(__file__).parent / "fixtures/im_mseg_dice.py"))["DiceLoss"]


def criterion(args):
    return SegmentationLoss(
        ce_weight=args.ce_weight, dice_weight=args.dice_weight,
        dice_class_weights=args.dice_class_weights,
        negative_slice_handling=args.negative_slice_handling,
        include_background=args.include_background,
        dice_normalization=args.dice_normalization,
    )


def test_default_cli_uses_complete_im_mseg_recipe():
    args, _ = parse_args([])
    assert args.no_val is True
    assert Path(args.list_dir).name == "myops380_no_val"
    assert (args.max_epochs, args.batch_size, args.accum_steps) == (300, 16, 1)
    assert (args.base_lr, args.weight_decay, args.clip_grad) == (0.0003333, 0.0001, 0.0)
    assert (args.ce_weight, args.dice_weight) == (0.2, 0.8)
    assert args.dice_class_weights == [0.5, 1, 1, 1]
    loss = criterion(args)
    assert loss.dice.batch_dice and loss.dice.include_background
    assert loss.dice.normalization == "classes"
    # Precision/workers were deliberately left as the user's existing settings.
    assert args.amp == "auto" and args.num_workers == 4
    assert args.pretrained is args.init_weights is None


def test_validation_opt_in_and_other_profiles_keep_separate_manifests():
    args, _ = parse_args(["--with-val"])
    assert args.no_val is False
    assert Path(args.list_dir).parts[-3:] == ("data", "processed", "splits")
    for profile in ("myopspp_unet2d_roi128_1mm",):
        args, _ = parse_args(["--dataset", profile])
        assert args.no_val is True
        assert profile in args.list_dir


@pytest.mark.parametrize("batch", [1, 3, 16])
@pytest.mark.parametrize("target_mode", ["random", "background", "microlesion", "one_positive_slice"])
def test_dice_and_segmentation_loss_values_and_gradients_match_reference(batch, target_mode):
    args, _ = parse_args([])
    rng = torch.Generator().manual_seed(42)
    logits = torch.randn(batch, 4, 13, 17, generator=rng, requires_grad=True)
    target = torch.randint(0, 4, (batch, 13, 17), generator=rng)
    if target_mode != "random":
        target.zero_()
    if target_mode == "microlesion":
        target[0, 3, 3] = 3
    elif target_mode == "one_positive_slice":
        target[0, 2:9, 3:8] = 2
    reference_logits = logits.detach().clone().requires_grad_()
    actual = criterion(args)(logits, target)
    expected_dice = REFERENCE_DICE(4)(reference_logits, target, softmax=True)
    expected = 0.2 * torch.nn.functional.cross_entropy(reference_logits, target) + 0.8 * expected_dice
    torch.testing.assert_close(actual["dice_loss"], expected_dice, rtol=0, atol=0)
    torch.testing.assert_close(actual["loss"], expected, rtol=0, atol=0)
    actual["loss"].backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=0, atol=0)


def test_batch_dice_does_not_turn_accumulation_into_a_physical_large_batch():
    loss = criterion(parse_args([])[0])
    with pytest.raises(ValueError, match="accum_steps=1"):
        loss.validate_accumulation(2)
    with pytest.raises(SystemExit):
        parse_args(["--accum-steps", "2"])


def test_optimizer_decay_applies_to_bias_norm_and_all_trainable_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    reference_model = deepcopy(model)
    actual = build_optimizer(model, 0.0003333, 0.0001)
    expected = torch.optim.AdamW(reference_model.parameters(), lr=0.0003333, weight_decay=0.0001)
    assert len(actual.param_groups) == 1
    assert {id(p) for p in actual.param_groups[0]["params"]} == {id(p) for p in model.parameters()}
    assert actual.param_groups[0]["weight_decay"] == 0.0001
    for _ in range(3):
        for a, b in zip(model.parameters(), reference_model.parameters()):
            a.grad = torch.ones_like(a)
            b.grad = torch.ones_like(b)
        actual.step()
        expected.step()
    for a, b in zip(model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_lr_used_on_every_update_and_resume_matches_original_loop():
    model = torch.nn.Linear(1, 1)
    optimizer = build_optimizer(model, 0.0003333, 0.0001)
    scheduler = build_scheduler(optimizer, total_updates=7)
    lr = 0.0003333
    for iteration in range(7):
        assert optimizer.param_groups[0]["lr"] == lr
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        optimizer.step()
        scheduler.step()
        # Verbatim LR update rule from I-MMSeg/trainer.py, after optimizer.step.
        lr = 0.0003333 * (1.0 - iteration / 7) ** 0.9
        assert optimizer.param_groups[0]["lr"] == lr
        assert scheduler.last_epoch == iteration + 1
        if iteration == 2:
            restored_optimizer = build_optimizer(model, 0.0003333, 0.0001)
            restored_scheduler = build_scheduler(restored_optimizer, total_updates=7)
            restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
            restored_scheduler.load_state_dict(deepcopy(scheduler.state_dict()))
            optimizer, scheduler = restored_optimizer, restored_scheduler
    assert lr > 0  # The reference does not reach zero after its final update.


def test_optimizer_and_dice_normalization_cannot_silently_resume_an_old_recipe():
    validate_optimization_checkpoint({"optimization_protocol": deepcopy(OPTIMIZATION_PROTOCOL)})
    with pytest.raises(ValueError, match="optimizer/LR scheduler"):
        validate_optimization_checkpoint({})
    args, _ = parse_args([])
    _validate_resume_loss_config(vars(args), args, saved_version=LOSS_PROTOCOL_VERSION)
    with pytest.raises(ValueError, match="loss protocol"):
        _validate_resume_loss_config(vars(args), args, saved_version=3)
    old = vars(args) | {"dice_normalization": "weights"}
    with pytest.raises(ValueError, match="normalization"):
        _validate_resume_loss_config(old, args, saved_version=3)


def test_legacy_weight_normalization_remains_explicitly_available():
    p = torch.zeros(1, 4, 4, 4)
    y = torch.zeros(1, 4, 4, dtype=torch.long)
    by_classes = DiceLoss(batch_dice=True, include_background=True, normalization="classes")
    by_weights = DiceLoss(batch_dice=True, include_background=True, normalization="weights")
    a, b = by_classes(p, y, [0.5, 1, 1, 1]), by_weights(p, y, [0.5, 1, 1, 1])
    torch.testing.assert_close(a * (4 / 3.5), b)
