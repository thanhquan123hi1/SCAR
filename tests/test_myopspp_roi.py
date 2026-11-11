"""Regression checks for ROI auxiliary-loss reductions."""
import pytest
import torch
import numpy as np

def test_sample_pathology_reduction_accumulates_with_uneven_positive_counts():
    from training.loss.losses import SegmentationLoss
    torch.manual_seed(1234)
    logits = torch.randn(4, 4, 4, 4, requires_grad=True)
    target = torch.zeros(4, 4, 4, dtype=torch.long)
    target[0, 1:3, 1:3] = 2
    target[2, 1:3, 1:3] = 3
    loss = SegmentationLoss(aar_weight=0.2, scar_weight=0.2, pathology_reduction="sample")
    full = loss(logits, target)
    grad = torch.autograd.grad(full["loss"], logits)[0]
    pieces = [loss(logits[i:i+1], target[i:i+1])["loss"] for i in range(4)]
    accumulated = sum(pieces) / 4
    np.testing.assert_allclose(accumulated.detach(), full["loss"].detach(), rtol=1e-6)
    torch.testing.assert_close(torch.autograd.grad(accumulated, logits)[0], grad)
    legacy = SegmentationLoss(aar_weight=0.2, scar_weight=0.2)(logits, target)
    torch.testing.assert_close(full["aar_loss"], legacy["aar_loss"] * 0.5)
    torch.testing.assert_close(full["scar_loss"], legacy["scar_loss"] * 0.25)


def test_resume_rejects_changed_reduction_and_accepts_legacy_missing_setting():
    from argparse import Namespace
    from training.trainer.trainer import _validate_resume_loss_config
    _validate_resume_loss_config({}, Namespace(pathology_reduction="positive"))
    with pytest.raises(ValueError, match="pathology_reduction"):
        _validate_resume_loss_config({}, Namespace(pathology_reduction="sample"))
