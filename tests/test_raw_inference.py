"""Benchmark inference must preserve the network's unfiltered predictions."""
import math

import numpy as np
import pytest
import torch

from training.evaluate import build_parser
from training.predict import predict_volume


class LabelLogitModel(torch.nn.Module):
    """A deterministic network that makes disconnected lesions easy to inspect."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, cine, lge, t2w):
        self.calls.append((cine.shape[0], self.training))
        labels = (cine[:, 0] * 3).round().long()
        logits = cine.new_full((cine.shape[0], 4, *cine.shape[-2:]), -4)
        return logits.scatter_(1, labels[:, None], 4) + self.anchor * 0


def test_single_pass_preserves_small_disconnected_lesions_and_model_mode():
    mask = np.zeros((32, 32, 3), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    mask[2, 2] = 3
    mask[29, 29] = 2
    images = [mask.astype(np.float32) / 3] * 3
    model = LabelLogitModel().train()

    prediction = predict_volume(model, images, img_size=32, batch_size=2, device="cpu")

    np.testing.assert_array_equal(prediction, mask)
    assert prediction.dtype == np.uint8
    assert len(model.calls) == math.ceil(mask.shape[2] / 2)
    assert all(not training for _, training in model.calls)
    assert model.training


@pytest.mark.parametrize("flag", ["--tta", "--no-tta", "--postprocess", "--no-postprocess"])
def test_evaluation_rejects_removed_prediction_modifiers(flag):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["--checkpoint", "best.pth", "--data-root", "cache", flag])
    assert error.value.code == 2


def test_prediction_api_rejects_test_time_augmentation():
    images = [np.zeros((32, 32, 1), dtype=np.float32)] * 3
    with pytest.raises(TypeError, match="tta"):
        predict_volume(LabelLogitModel(), images, device="cpu", tta=True)
