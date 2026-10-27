"""Model construction and compatibility with checkpoint metadata."""
from __future__ import annotations

from training.models.m2_plus_net import M2PlusNet, get_config, get_testing
from training.models.immseg import IMMSegNet, get_immseg_config


def validate_model_config(config):
    """Validate architecture and ablation settings before loading weights or running training."""
    architecture = str(config.get("architecture", "m2_plus")).lower().replace("-", "_")
    ablation = str(config.get("ablation", "M2-PLUS")).upper()

    if architecture in {"immseg", "i_mmseg"}:
        if ablation != "IMMSEG":
            raise ValueError(f"I-MMSeg architecture must use ablation IMMSEG; got {ablation!r}")
        return

    if architecture not in {"m2_plus", "m2plus", "cmspa_net"}:
        raise ValueError(f"Unsupported architecture: {architecture!r}. Expected 'm2_plus' or 'immseg'.")
    if architecture == "cmspa_net" and "ablation" not in config:
        raise ValueError("Legacy cmspa_net checkpoint must identify ablation M2-PLUS")
    if ablation != "M2-PLUS":
        raise ValueError(f"M2-Plus architecture requires ablation 'M2-PLUS'; received {ablation!r}")


def model_from_config(config, **kwargs):
    validate_model_config(config)
    architecture = str(config.get("architecture", "m2_plus")).lower().replace("-", "_")
    if architecture in {"immseg", "i_mmseg"}:
        return IMMSegNet(config=config, **kwargs)
    return M2PlusNet(config=config, **kwargs)


def build_model(model_name="m2_plus", **kwargs):
    """Build model instance by name (m2_plus or immseg)."""
    norm_name = str(model_name).lower().replace("-", "_")
    config = kwargs.get("config", {})
    selected = dict(config)
    selected["architecture"] = model_name
    if "ablation" in kwargs:
        selected["ablation"] = kwargs["ablation"]
    elif norm_name in {"immseg", "i_mmseg"}:
        selected["ablation"] = "IMMSEG"
    validate_model_config(selected)
    if config:
        validate_model_config(config)

    if norm_name in {"immseg", "i_mmseg"}:
        return IMMSegNet(**kwargs)
    return M2PlusNet(**kwargs)


__all__ = [
    "M2PlusNet",
    "IMMSegNet",
    "get_config",
    "get_testing",
    "get_immseg_config",
    "build_model",
    "model_from_config",
    "validate_model_config",
]
