from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from training.loss.losses import SegmentationLoss
from training.models import IMMSegNet, build_model, get_immseg_config, model_from_config, validate_model_config
from training.predict import predict_volume
from training.train import parse_args


def test_immseg_config_parsing():
    args, merged = parse_args(["--config", "training/config/models/immseg.yaml", "--ablation", "IMMSEG"])
    assert args.ablation == "IMMSEG"
    assert merged["model"]["architecture"] == "immseg"
    assert merged["model"]["model_name"] == "I-MMSeg"
    assert merged["model"]["num_classes"] == 4
    # Official I-MMSeg native loss configuration: 0.2 CE + 0.8 SoftDice with background weight 0.5
    assert args.ce_weight == 0.2
    assert args.dice_weight == 0.8
    assert args.dice_class_weights == [0.5, 1.0, 1.0, 1.0]



def test_immseg_validation_rules():
    # Valid I-MMSeg config
    validate_model_config({"architecture": "immseg", "ablation": "IMMSEG"})
    validate_model_config({"architecture": "i_mmseg", "ablation": "IMMSEG"})

    # Invalid ablation
    try:
        validate_model_config({"architecture": "immseg", "ablation": "M2-PLUS"})
        raise AssertionError("Expected ValueError for invalid ablation")
    except ValueError as e:
        assert "I-MMSeg architecture must use ablation IMMSEG" in str(e)

    # Unknown architecture
    try:
        validate_model_config({"architecture": "unknown", "ablation": "IMMSEG"})
        raise AssertionError("Expected ValueError for unknown architecture")
    except ValueError as e:
        assert "Unsupported architecture" in str(e)


def test_immseg_instantiation():
    cfg = get_immseg_config()
    model = model_from_config(cfg)
    assert isinstance(model, IMMSegNet)
    assert model.num_classes == 4
    assert hasattr(model, "class_embedding")
    assert hasattr(model, "modal_embedding")
    assert model.class_embedding.shape == (4, 512)
    assert model.modal_embedding.shape == (3, 512)
    assert hasattr(model, "config")
    assert model.config.architecture == "immseg"


def test_immseg_forward_separate_and_concat():
    cfg = get_immseg_config()
    model = IMMSegNet(config=cfg, img_size=128, num_classes=4)
    model.eval()

    c0 = torch.randn(2, 1, 128, 128)
    lge = torch.randn(2, 1, 128, 128)
    t2 = torch.randn(2, 1, 128, 128)

    with torch.no_grad():
        # Separate modalities
        out1 = model(c0, lge, t2)
        assert out1.shape == (2, 4, 128, 128)

        # Concatenated 3-channel input
        concat = torch.cat([c0, lge, t2], dim=1)
        out2 = model(concat)
        assert out2.shape == (2, 4, 128, 128)
        assert torch.allclose(out1, out2, atol=1e-5)


def test_immseg_gradients_flow():
    cfg = get_immseg_config()
    model = IMMSegNet(config=cfg, img_size=128, num_classes=4)
    model.train()

    c0 = torch.randn(2, 1, 128, 128)
    lge = torch.randn(2, 1, 128, 128)
    t2 = torch.randn(2, 1, 128, 128)
    target = torch.randint(0, 4, (2, 128, 128))

    criterion = SegmentationLoss(
        n_classes=4,
        ce_weight=0.2,
        dice_weight=0.8,
        dice_class_weights=[0.5, 1.0, 1.0, 1.0],
    )
    logits = model(c0, lge, t2)
    loss = criterion(logits, target)["loss"]

    assert torch.isfinite(loss)
    loss.backward()

    # Check gradients in encoders, fusion, and dynamic prediction head
    assert model.transformer1.embeddings.hybrid_model.root.conv.weight.grad is not None
    assert model.transformer1.embeddings.hybrid_model.root.conv.weight.grad.abs().sum() > 0
    assert model.mix_cine.fusion_proj.weight.grad is not None
    assert model.mix_cine.fusion_proj.weight.grad.abs().sum() > 0
    assert model.forward_prediction_head.controller.weight.grad is not None
    assert model.forward_prediction_head.controller.weight.grad.abs().sum() > 0


def test_immseg_predict_volume():
    cfg = get_immseg_config()
    model = IMMSegNet(config=cfg, img_size=128, num_classes=4)
    model.eval()

    images = [np.random.randn(128, 128, 3).astype(np.float32) for _ in range(3)]
    pred = predict_volume(model, images, img_size=128, batch_size=2)
    assert pred.shape == (128, 128, 3)
    assert set(np.unique(pred)).issubset({0, 1, 2, 3})


if __name__ == "__main__":
    test_immseg_config_parsing()
    test_immseg_validation_rules()
    test_immseg_instantiation()
    test_immseg_forward_separate_and_concat()
    test_immseg_gradients_flow()
    test_immseg_predict_volume()
    print("ALL I-MMSEG TESTS PASSED!")
