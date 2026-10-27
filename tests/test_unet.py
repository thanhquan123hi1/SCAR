import numpy as np
import torch

from training.loss.losses import SegmentationLoss
from training.models import UNet2D, build_model, get_unet_config, model_from_config, validate_model_config
from training.predict import predict_volume
from training.train import parse_args


def test_unet_config_parsing():
    args, merged = parse_args(["--config", "training/config/models/unet.yaml", "--ablation", "UNET"])
    assert args.ablation == "UNET"
    assert merged["model"]["architecture"] == "unet"
    assert merged["model"]["model_name"] == "U-Net"
    assert merged["model"]["num_classes"] == 4


def test_unet_validation_rules():
    # Valid U-Net config
    validate_model_config({"architecture": "unet", "ablation": "UNET"})
    validate_model_config({"architecture": "u_net", "ablation": "UNET"})

    # Invalid ablation
    try:
        validate_model_config({"architecture": "unet", "ablation": "M2-PLUS"})
        raise AssertionError("Expected ValueError for invalid ablation")
    except ValueError as e:
        assert "U-Net architecture must use ablation UNET" in str(e)

    # Unknown architecture
    try:
        validate_model_config({"architecture": "unknown", "ablation": "UNET"})
        raise AssertionError("Expected ValueError for unknown architecture")
    except ValueError as e:
        assert "Unsupported architecture" in str(e)


def test_unet_instantiation():
    cfg = get_unet_config()
    model = model_from_config(cfg)
    assert isinstance(model, UNet2D)
    assert model.n_channels == 3
    assert model.n_classes == 4
    assert hasattr(model, "config")
    assert model.config.architecture == "unet"


def test_unet_forward_separate_and_concat():
    model = UNet2D(in_channels=3, num_classes=4, base_channels=16)  # smaller for fast unit test
    lge = torch.randn(2, 1, 64, 64)
    c0 = torch.randn(2, 1, 64, 64)
    t2 = torch.randn(2, 1, 64, 64)

    # Separate modalities
    out1 = model(lge, c0, t2)
    assert out1.shape == (2, 4, 64, 64)

    # Concatenated 3-channel input
    concat = torch.cat([lge, c0, t2], dim=1)
    out2 = model(concat)
    assert out2.shape == (2, 4, 64, 64)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_unet_gradients_flow():
    model = UNet2D(in_channels=3, num_classes=4, base_channels=16)
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 4, (2, 64, 64))

    criterion = SegmentationLoss(n_classes=4, ce_weight=0.5, dice_weight=0.5)
    logits = model(x)
    loss = criterion(logits, target)["loss"]

    assert torch.isfinite(loss)
    loss.backward()

    # Check gradients in input convolution, bottleneck, and output convolution
    assert model.inc.double_conv[0].weight.grad is not None
    assert model.inc.double_conv[0].weight.grad.abs().sum() > 0
    assert model.down4.maxpool_conv[1].double_conv[0].weight.grad is not None
    assert model.down4.maxpool_conv[1].double_conv[0].weight.grad.abs().sum() > 0
    assert model.outc.conv.weight.grad is not None
    assert model.outc.conv.weight.grad.abs().sum() > 0


def test_unet_predict_volume():
    model = UNet2D(in_channels=3, num_classes=4, base_channels=16)
    images = [np.random.randn(64, 64, 5).astype(np.float32) for _ in range(3)]
    pred = predict_volume(model, images, img_size=64, batch_size=2)
    assert pred.shape == (64, 64, 5)
    assert set(np.unique(pred)).issubset({0, 1, 2, 3})


if __name__ == "__main__":
    test_unet_config_parsing()
    test_unet_validation_rules()
    test_unet_instantiation()
    test_unet_forward_separate_and_concat()
    test_unet_gradients_flow()
    test_unet_predict_volume()
    print("ALL TESTS PASSED!")
