import torch

from training.models import CMSPANet, build_model, get_testing, model_from_config
from training.train import parse_args


def test_m2_plus_config_selects_model():
    args, merged = parse_args(["--config", "training/config/models/m2_plus.yaml"])
    assert args.ablation.upper() == "M2-PLUS"
    assert merged["model"]["model_name"] == "M2-Plus"

    config = get_testing()
    config.ablation = args.ablation
    model = model_from_config(config, img_size=32)
    assert isinstance(model, CMSPANet)
    assert model.ablation == "M2-PLUS"
    assert type(model.cross_fusion).__name__ == "M2Plus_Fusion"


def test_m2_plus_separate_attention_streams_receive_gradients():
    config = get_testing()
    model = build_model("cmspa_net", config=config, img_size=32, ablation="M2-Plus")
    cine, psir, t2w = (torch.randn(2, 1, 32, 32) for _ in range(3))

    logits = model(cine, psir, t2w)
    assert logits.shape == (2, 4, 32, 32)
    logits.square().mean().backward()
    assert model.cross_fusion.mha_scar.in_proj_weight.grad.abs().sum() > 0
    assert model.cross_fusion.mha_edema.in_proj_weight.grad.abs().sum() > 0
