import torch

from training.loss.losses import SegmentationLoss

from training.models import M2PlusNet, build_model, get_testing, model_from_config
from training.train import parse_args


def test_m2_plus_config_selects_model():
    args, merged = parse_args(["--config", "training/config/models/m2_plus.yaml"])
    assert args.ablation.upper() == "M2-PLUS"
    assert merged["model"]["model_name"] == "M2-Plus"

    config = get_testing()
    config.ablation = args.ablation
    model = model_from_config(config, img_size=32)
    assert isinstance(model, M2PlusNet)
    assert model.ablation == "M2-PLUS"
    assert type(model.cross_fusion).__name__ == "M2Plus_Fusion"


def test_m2_plus_separate_attention_streams_receive_gradients():
    config = get_testing()
    model = build_model("m2_plus", config=config, img_size=32, ablation="M2-Plus")
    cine, psir, t2w = (torch.randn(2, 1, 32, 32) for _ in range(3))

    logits = model(cine, psir, t2w)
    assert logits.shape == (2, 4, 32, 32)
    logits.square().mean().backward()
    assert model.cross_fusion.mha_scar.in_proj_weight.grad.abs().sum() > 0
    assert model.cross_fusion.mha_edema.in_proj_weight.grad.abs().sum() > 0


def test_segmentation_loss_conditional_aar():
    criterion = SegmentationLoss(n_classes=4, aar_weight=0.25)
    logits = torch.randn(2, 4, 32, 32)
    # Target without pathology
    target_clean = torch.randint(0, 2, (2, 32, 32))
    loss_clean = criterion(logits, target_clean)
    assert loss_clean["aar_loss"].item() == 0.0
    assert torch.isfinite(loss_clean["loss"])

    # Target with pathology
    target_patho = torch.randint(0, 4, (2, 32, 32))
    target_patho[0, 5, 5] = 2
    target_patho[1, 10, 10] = 3
    loss_patho = criterion(logits, target_patho)
    assert loss_patho["aar_loss"].item() > 0.0
    assert torch.isfinite(loss_patho["loss"])
