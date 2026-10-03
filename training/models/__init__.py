"""Model architectures and model registry."""
from __future__ import annotations

from functools import partial
from typing import Callable

from torch import nn

from training.models.backbones.resnet_v2 import PreActBottleneck, ResNetV2, StdConv2d
from training.models.cmspa_net import (
    CONFIGS,
    CMSPANet,
    Embeddings,
    Transformer,
    VisionTransformer,
    get_config,
    get_r50_b16_config,
    get_r50_l16_config,
    get_testing,
)
from training.models.modules.cmspa import CMSPA_Fusion
from training.models.modules.decoder import DecoderCup, SegmentationHead
from training.models.modules.fusion import ConcatFusion, CrossAttention_Fusion, Fusion_Embed
from training.models.modules.m2_plus import M2Plus_Fusion
from training.models.modules.sspanet import SSPANet_Block

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "cmspa_net": CMSPANet,
    "cmspa": CMSPANet,
    "vision_transformer": CMSPANet,
    "cross_attn_baseline": partial(CMSPANet, ablation="M2"),
    "m2_plus": partial(CMSPANet, ablation="M2-PLUS"),
    "m2plus": partial(CMSPANet, ablation="M2-PLUS"),
    "m2_pro": partial(CMSPANet, ablation="M2-PRO"),
    "m2pro": partial(CMSPANet, ablation="M2-PRO"),
    "m2_max": partial(CMSPANet, ablation="M2-MAX"),
    "m2max": partial(CMSPANet, ablation="M2-MAX"),
    "m2_max_pro": partial(CMSPANet, ablation="M2-MAX-PRO"),
    "m2maxpro": partial(CMSPANet, ablation="M2-MAX-PRO"),
    "m2_max_v4": partial(CMSPANet, ablation="M2-MAX-V4"),
    "m2maxv4": partial(CMSPANet, ablation="M2-MAX-V4"),
    "m2_max_v5": partial(CMSPANet, ablation="M2-MAX-V5"),
    "m2maxv5": partial(CMSPANet, ablation="M2-MAX-V5"),
    "m2_max_v6": partial(CMSPANet, ablation="M2-MAX-V6"),
    "m2maxv6": partial(CMSPANet, ablation="M2-MAX-V6"),
    "h_cmspa": partial(CMSPANet, ablation="H-CMSPA"),
    "hcmspa": partial(CMSPANet, ablation="H-CMSPA"),
}


def model_from_config(config, **kwargs):
    """Old checkpoints without an architecture field remain CMSPANet."""
    architecture = config.get("architecture", "cmspa_net")
    return build_model(architecture, config=config, **kwargs)


def build_model(model_name: str, **kwargs) -> nn.Module:
    """Instantiate a registered model by name."""
    name = model_name.lower().replace("-", "_")
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    expected = {"cross_attn_baseline": "M2", "m2_plus": "M2-PLUS", "m2plus": "M2-PLUS"}.get(name)
    if expected is not None and kwargs.get("ablation", expected).upper() != expected:
        raise ValueError(f"Model {model_name!r} requires ablation {expected}")
    return MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "model_from_config",
    "CMSPANet",
    "VisionTransformer",
    "CONFIGS",
    "get_config",
    "get_r50_b16_config",
    "get_r50_l16_config",
    "get_testing",
    "Embeddings",
    "Transformer",
    "ResNetV2",
    "StdConv2d",
    "PreActBottleneck",
    "SSPANet_Block",
    "CMSPA_Fusion",
    "ConcatFusion",
    "CrossAttention_Fusion",
    "M2Plus_Fusion",
    "Fusion_Embed",
    "DecoderCup",
    "SegmentationHead",
    "MODEL_REGISTRY",
    "build_model",
]
