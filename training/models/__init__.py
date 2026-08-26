"""Model registry and factory for LGE segmentation."""

from __future__ import annotations

from typing import Any
import torch.nn as nn

from training.models.unet_3d import UNet3D
from training.models.unet_2d import UNet2D
from training.models.resunet_plus_plus import ResUNetPlusPlus2D, MultiDatasetResUNetPlusPlus

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "unet_3d": UNet3D,
    "unet_2d": UNet2D,
    "resunet_plus_plus": ResUNetPlusPlus2D,
    "resunet_plus_plus_2d": ResUNetPlusPlus2D,
    "resunet_plus_plus_multihead": MultiDatasetResUNetPlusPlus,
}


def build_model(model_name: str, **kwargs: Any) -> nn.Module:
    """Build model instance by name from registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)

