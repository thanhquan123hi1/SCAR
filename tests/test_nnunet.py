from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from training.models import build_model, model_from_config
from training.models.nnunet import NNUNet2D
from training.train import parse_args


def test_nnunet_config_and_instantiation():
    print("Testing config parsing and model instantiation...")
    args, merged = parse_args(["--config", "training/config/models/nnunet.yaml"])
    assert args.ablation == "NNUNET", f"Expected NNUNET, got {args.ablation}"
    assert merged["model"]["architecture"] == "nnunet"
    assert merged["model"]["model_name"] == "nnU-Net"

    model = model_from_config(merged["model"], img_size=128, num_classes=4)
    assert isinstance(model, NNUNet2D)
    assert model.num_classes == 4
    print(" Config parsing and instantiation verified!")


def test_nnunet_forward_backward():
    print("Testing forward pass (3 modalities: cine, psir, t2w) and backward pass...")
    model = build_model("nnunet", img_size=128, num_classes=4)
    cine, psir, t2w = (torch.randn(2, 1, 128, 128) for _ in range(3))

    # Test multi-tensor forward (Contract used by Trainer and predict_volume)
    logits = model(cine, psir, t2w)
    assert logits.shape == (2, 4, 128, 128), f"Expected shape (2, 4, 128, 128), got {logits.shape}"

    # Test single-tensor concatenated forward
    concat_input = torch.cat([cine, psir, t2w], dim=1)
    logits_concat = model(concat_input)
    assert logits_concat.shape == (2, 4, 128, 128), f"Expected shape (2, 4, 128, 128), got {logits_concat.shape}"

    # Test gradient backward pass
    loss = logits.square().mean()
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "Gradients did not flow through the network"
    print(" Forward & Backward passes verified successfully!")


if __name__ == "__main__":
    test_nnunet_config_and_instantiation()
    test_nnunet_forward_backward()
    print("\n>>> ALL NNUNET TESTS PASSED PERFECTLY! <<<")
