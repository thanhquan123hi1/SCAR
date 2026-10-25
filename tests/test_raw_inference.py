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
        build_parser().parse_args(["--checkpoint", "best.pt", "--data-root", "cache", flag])
    assert error.value.code == 2


def test_prediction_api_rejects_test_time_augmentation():
    images = [np.zeros((32, 32, 1), dtype=np.float32)] * 3
    with pytest.raises(TypeError, match="tta"):
        predict_volume(LabelLogitModel(), images, device="cpu", tta=True)


def test_raw_nifti_prediction_loads_legacy_m2_plus_and_preserves_geometry(tmp_path):
    import json
    import nibabel as nib
    from training.dataset.data_contract import CLASS_NAMES
    from training.models import M2PlusNet, get_testing
    from training.predict import main as predict

    torch.set_num_threads(2)
    model = M2PlusNet(get_testing(), img_size=32).eval()
    config = model.config.to_dict() | {"architecture": "cmspa_net", "ablation": "M2-Plus"}
    checkpoint = tmp_path / "legacy_m2_plus.pt"
    torch.save(dict(format_version=1, class_names=CLASS_NAMES, model=model.state_dict(),
                    model_config=config, args=dict(img_size=32, dataset_id="myopspp_bc80"),
                    data_provenance=dict(metadata=dict(normalization="unit255"))), checkpoint)
    affine = np.diag([1.5, 2.0, 7.0, 1.0])
    arrays, paths = [], []
    for i, modality in enumerate(("cine", "lge", "t2w")):
        array = np.random.default_rng(i).uniform(0, 255, (40, 48, 2)).astype(np.float32)
        paths.append(tmp_path / f"{modality}.nii.gz")
        volume = nib.Nifti1Image(array, affine)
        volume.header.set_xyzt_units("mm")
        nib.save(volume, paths[-1])
        arrays.append(array / 255)
    output = tmp_path / "prediction.nii.gz"
    argv = ["--checkpoint", str(checkpoint), "--cine", str(paths[0]), "--lge", str(paths[1]),
            "--t2w", str(paths[2]), "--normalization", "unit255", "--output", str(output),
            "--device", "cpu", "--amp", "none", "--cpu-threads", "2", "--batch-size", "2"]
    assert predict(argv) == output
    prediction = nib.load(output)
    np.testing.assert_array_equal(prediction.get_fdata(),
                                  predict_volume(model, arrays, img_size=32, batch_size=2, device="cpu"))
    np.testing.assert_array_equal(prediction.affine, affine)
    assert prediction.shape == (40, 48, 2)
    assert prediction.get_data_dtype() == np.dtype("uint8")
    assert prediction.header.get_xyzt_units()[0] == "mm"
    metadata = json.loads((tmp_path / "prediction.nii.gz.json").read_text())
    assert metadata["class_names"] == list(CLASS_NAMES)
    with pytest.raises(FileExistsError):
        predict(argv)

    roi_output = tmp_path / "roi_prediction.nii.gz"
    payload = torch.load(checkpoint, weights_only=True)
    payload["args"]["dataset_id"] = "myopspp_roi128_80"
    torch.save(payload, checkpoint)
    argv[argv.index("--output") + 1] = str(roi_output)
    with pytest.raises(ValueError, match="ROI checkpoint"):
        predict(argv)
    assert not roi_output.exists()
