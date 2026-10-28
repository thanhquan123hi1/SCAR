"""Tests for Canonical TransUNet Baseline in SCAR."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ml_collections import ConfigDict

from training.models.transunet import TransUNet2D, get_transunet_config
from training.models import model_from_config, model_architecture, validate_model_config


def test_transunet_config():
    config = get_transunet_config()
    assert config.architecture == "transunet"
    assert config.ablation == "TRANSUNET"
    assert config.num_classes == 4
    assert config.in_channels == 3
    assert config.img_size == 128
    assert config.hidden_size == 768
    assert config.transformer.num_layers == 12
    assert config.transformer.num_heads == 12


def test_transunet_model_architecture():
    config = {"architecture": "transunet", "ablation": "TRANSUNET"}
    assert model_architecture(config) == "transunet"
    validate_model_config(config)


def test_transunet_forward_concatenated():
    config = get_transunet_config()
    # Test on CPU with 1 sample
    model = TransUNet2D(config=config, img_size=128, num_classes=4)
    model.eval()

    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 4, 128, 128)


def test_transunet_forward_three_modalities():
    config = get_transunet_config()
    model = TransUNet2D(config=config, img_size=128, num_classes=4)
    model.eval()

    m1 = torch.randn(2, 1, 128, 128)
    m2 = torch.randn(2, 1, 128, 128)
    m3 = torch.randn(2, 1, 128, 128)

    with torch.no_grad():
        out = model(m1, m2, m3)
    assert out.shape == (2, 4, 128, 128)


def test_transunet_gradient_flow():
    config = get_transunet_config()
    model = TransUNet2D(config=config, img_size=128, num_classes=4)
    model.train()

    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Check gradients exist in patch embedding and classification head
    assert model.transformer.embeddings.patch_embeddings.weight.grad is not None
    assert model.segmentation_head[0].weight.grad is not None


def test_transunet_model_from_config():
    config = {
        "architecture": "transunet",
        "ablation": "TRANSUNET",
        "num_classes": 4,
        "img_size": 128,
    }
    model = model_from_config(config)
    assert isinstance(model, TransUNet2D)
    assert model.num_classes == 4


if __name__ == "__main__":
    print("Testing TransUNet config...")
    test_transunet_config()
    print("Testing TransUNet architecture registration...")
    test_transunet_model_architecture()
    print("Testing TransUNet forward (concatenated)...")
    test_transunet_forward_concatenated()
    print("Testing TransUNet forward (3 modalities)...")
    test_transunet_forward_three_modalities()
    print("Testing TransUNet gradient flow...")
    test_transunet_gradient_flow()
    print("Testing TransUNet model_from_config...")
    test_transunet_model_from_config()
    print("All TransUNet tests passed 100%!")
