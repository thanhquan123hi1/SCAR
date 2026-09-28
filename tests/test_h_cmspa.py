import torch
from training.models import CMSPANet, build_model, get_testing, model_from_config
from training.train import parse_args


def test_h_cmspa_config_selects_model():
    args, merged = parse_args(["--config", "training/config/models/h_cmspa.yaml"])
    assert args.ablation.upper() in {"H-CMSPA", "HCMSPA"}
    assert merged["model"]["model_name"] == "H-CMSPA"

    config = get_testing()
    config.ablation = args.ablation
    model = model_from_config(config, img_size=32)
    assert isinstance(model, CMSPANet)
    assert model.ablation in {"H-CMSPA", "HCMSPA"}
    assert type(model.cross_fusion).__name__ == "H_CMSPA_Fusion"


def test_h_cmspa_separate_attention_streams_and_gating_receive_gradients():
    config = get_testing()
    model = build_model("cmspa_net", config=config, img_size=32, ablation="H-CMSPA")
    cine, psir, t2w = (torch.randn(2, 1, 32, 32) for _ in range(3))

    logits = model(cine, psir, t2w)
    assert logits.shape == (2, 4, 32, 32)
    logits.square().mean().backward()
    assert model.cross_fusion.mha_scar.in_proj_weight.grad.abs().sum() > 0
    assert model.cross_fusion.mha_edema.in_proj_weight.grad.abs().sum() > 0
    assert model.cross_fusion.conv_strip[0].weight.grad.abs().sum() > 0
    assert model.cross_fusion.conv_patho[0].weight.grad.abs().sum() > 0
