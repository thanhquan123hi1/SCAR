"""M2-Plus & nnU-Net baseline construction and compatibility with checkpoint metadata."""
from __future__ import annotations

from training.models.m2_plus_net import (
    M2PlusNet, LightweightM2PlusNet, get_config, get_testing,
    get_lightweight_config, get_lightweight_testing,
)
from training.models.nnunet import NNUNet2D
from training.models.modules.sspanet import LEGACY_SSPA_GATE, SSPA_GATE_MODES
from training.models.modules.m2_plus import LEGACY_FUSION, SPATIAL_FUSION, FUSION_VERSIONS


def model_architecture(config):
    architecture = str(config.get("architecture", "m2_plus")).lower().replace("-", "_")
    if architecture in {"nnunet", "nn_unet"}:
        return "nnunet"
    if (architecture in {"lightweight_m2_plus", "lightweight"}
            or str(config.get("ablation", "")).upper() in {"LIGHTWEIGHT_M2_PLUS", "LIGHTWEIGHT"}):
        return "lightweight_m2_plus"
    return "m2_plus"


def validate_model_config(config):
    """Reject retired architectures before loading weights or modifying a run."""
    architecture = str(config.get("architecture", "m2_plus")).lower().replace("-", "_")
    if architecture in {"nnunet", "nn_unet"}:
        ablation = str(config.get("ablation", "NNUNET")).upper()
        if ablation != "NNUNET":
            raise ValueError(f"nnU-Net requires ablation 'NNUNET'; received {ablation!r}")
        return
    if architecture not in {"m2_plus", "m2plus", "cmspa_net", "lightweight_m2_plus", "lightweight"}:
        raise ValueError(f"Only M2-Plus and nnU-Net are supported; received architecture {architecture!r}")
    # cmspa_net was shared by several architectures. Its ablation must identify
    # M2-Plus explicitly rather than silently interpreting an old model as M2-Plus.
    if architecture == "cmspa_net" and "ablation" not in config:
        raise ValueError("Legacy cmspa_net checkpoint must identify ablation M2-PLUS")
    ablation = str(config.get("ablation", "M2-PLUS")).upper()
    if ablation not in {"M2-PLUS", "LIGHTWEIGHT_M2_PLUS", "LIGHTWEIGHT"}:
        raise ValueError(f"Only M2-Plus is supported; received ablation {config.get('ablation')!r}")
    if config.get("sspanet_gate", LEGACY_SSPA_GATE) not in SSPA_GATE_MODES:
        raise ValueError(f"Unknown SSPANet gate {config.get('sspanet_gate')!r}; expected {SSPA_GATE_MODES}")
    if config.get("fusion_version", LEGACY_FUSION) not in FUSION_VERSIONS:
        raise ValueError(f"Unknown fusion version {config.get('fusion_version')!r}")
    for key in ("latent_dim", "sspanet_reduction"):
        if key in config and (type(config[key]) is not int or config[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    if "use_pos_enc" in config and not isinstance(config["use_pos_enc"], bool):
        raise ValueError("use_pos_enc must be a boolean")


def checkpoint_model_config(config, state_dict):
    """Recover omitted fusion metadata from saved tensors, without changing weights."""
    validate_model_config(config)
    if model_architecture(config) == "nnunet":
        return dict(config)
    resolved = dict(config)
    prefix = "cross_attention." if model_architecture(config) == "lightweight_m2_plus" else "cross_fusion."
    projection = state_dict.get(prefix + "proj_in.0.weight")
    legacy_projection = state_dict.get(prefix + "proj.0.weight")
    inferred = SPATIAL_FUSION if projection is not None else LEGACY_FUSION if legacy_projection is not None else None
    if inferred is not None:
        if "fusion_version" in resolved and resolved["fusion_version"] != inferred:
            raise ValueError("Checkpoint fusion_version disagrees with its saved weights")
        resolved["fusion_version"] = inferred
        if projection is not None:
            latent_dim = projection.shape[0]
            if "latent_dim" in resolved and resolved["latent_dim"] != latent_dim:
                raise ValueError("Checkpoint latent_dim disagrees with its saved weights")
            resolved["latent_dim"] = latent_dim
            # Spatial fusion before explicit metadata always enabled PE.
            resolved.setdefault("use_pos_enc", True)
        else:
            resolved["latent_dim"] = legacy_projection.shape[1] // 3
            resolved["use_pos_enc"] = False
    return resolved


def model_from_config(config, **kwargs):
    selected = dict(config)
    if kwargs.get("ablation") is not None:
        selected["ablation"] = kwargs["ablation"]
    validate_model_config(selected)
    architecture = model_architecture(selected)
    cls = {"nnunet": NNUNet2D, "lightweight_m2_plus": LightweightM2PlusNet, "m2_plus": M2PlusNet}[architecture]
    return cls(config=selected, **kwargs)


def build_model(model_name="m2_plus", **kwargs):
    """Build M2-Plus or nnU-Net; cmspa_net is accepted only with explicit M2-Plus metadata."""
    config = kwargs.get("config", {})
    selected = dict(config)
    selected["architecture"] = model_name
    if "ablation" in kwargs:
        selected["ablation"] = kwargs["ablation"]
    validate_model_config(selected)
    if not config:
        if model_architecture(selected) == "lightweight_m2_plus":
            config = get_lightweight_config().to_dict()
        elif model_architecture(selected) == "m2_plus":
            config = get_config().to_dict()
    selected = dict(config) | {"architecture": model_name}
    kwargs.pop("config", None)
    return model_from_config(selected, **kwargs)


__all__ = ["M2PlusNet", "LightweightM2PlusNet", "NNUNet2D", "get_config", "get_testing", "build_model",
           "get_lightweight_config", "get_lightweight_testing", "model_from_config",
           "model_architecture", "checkpoint_model_config", "validate_model_config"]
