"""M2-Plus construction and compatibility with M2-Plus checkpoint metadata."""
from __future__ import annotations

from training.models.m2_plus_net import M2PlusNet, get_config, get_testing


def validate_model_config(config):
    """Reject retired architectures before loading weights or modifying a run."""
    architecture = str(config.get("architecture", "m2_plus")).lower().replace("-", "_")
    if architecture not in {"m2_plus", "m2plus", "cmspa_net"}:
        raise ValueError(f"Only M2-Plus is supported; received architecture {architecture!r}")
    # cmspa_net was shared by several architectures. Its ablation must identify
    # M2-Plus explicitly rather than silently interpreting an old model as M2-Plus.
    if architecture == "cmspa_net" and "ablation" not in config:
        raise ValueError("Legacy cmspa_net checkpoint must identify ablation M2-PLUS")
    if str(config.get("ablation", "M2-PLUS")).upper() != "M2-PLUS":
        raise ValueError(f"Only M2-Plus is supported; received ablation {config.get('ablation')!r}")


def model_from_config(config, **kwargs):
    validate_model_config(config)
    return M2PlusNet(config=config, **kwargs)


def build_model(model_name="m2_plus", **kwargs):
    """Build M2-Plus; cmspa_net is accepted only with explicit M2-Plus metadata."""
    config = kwargs.get("config", {})
    selected = dict(config)
    selected["architecture"] = model_name
    if "ablation" in kwargs:
        selected["ablation"] = kwargs["ablation"]
    validate_model_config(selected)
    if config:
        validate_model_config(config)
    return M2PlusNet(**kwargs)


__all__ = ["M2PlusNet", "get_config", "get_testing", "build_model",
           "model_from_config", "validate_model_config"]
