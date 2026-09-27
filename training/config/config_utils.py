"""Configuration utilities: YAML merging, type coercion, and run directory generation."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

YAML_KEY_TO_DEST: dict[str, str] = {
    "data.sampler": "sampler",
    "data.rare_boost": "rare_boost",
    "data.foreground_boost": "foreground_boost",
    "train.epochs": "max_epochs",
    "train.max_epochs": "max_epochs",
    "train.batch_size": "batch_size",
    "train.accum_steps": "accum_steps",
    "train.lr": "base_lr",
    "train.base_lr": "base_lr",
    "train.weight_decay": "weight_decay",
    "train.clip_grad": "clip_grad",
    "train.patience": "patience",
    "train.min_delta": "min_delta",
    "train.save_every": "save_every",
    "train.log_every": "log_every",
    "train.epochs_per_run": "epochs_per_run",
    "train.tensorboard": "tensorboard",
    "train.resume": "resume",
    "train.pretrained": "pretrained",
    "loss.ce_weight": "ce_weight",
    "loss.dice_weight": "dice_weight",
    "data.data_root": "data_root",
    "data.list_dir": "list_dir",
    "data.val_fraction": "val_fraction",
    "data.label_order": "label_order",
    "data.num_workers": "num_workers",
    "data.pin_memory": "pin_memory",
    "model.ablation": "ablation",
    "model.img_size": "img_size",
    "model.gradient_checkpointing": "gradient_checkpointing",
    "outputs.dir": "output_dir",
    "outputs.output_dir": "output_dir",
    "outputs.run_root": "run_root",
    "outputs.run_id": "run_id",
    "seed": "seed",
    "deterministic": "deterministic",
    "device": "device",
    "amp": "amp",
    "cpu_threads": "cpu_threads",
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base without mutating either input."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def flatten_config(config: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten hierarchical YAML dict into a key-value mapping."""
    items: list[tuple[str, Any]] = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


MODEL_KEYS = {
    "model.architecture",
    "model.model_name", "model.num_classes", "model.fused_channels",
    "model.cross_attention_heads", "model.resnet.num_layers", "model.resnet.width_factor",
    "model.decoder_channels", "model.skip_channels", "model.n_skip",
    "model.activation", "model.classifier", "model.transformer.dropout_rate",
}


def coerce_config_to_parser_types(flat_config: dict[str, Any], parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Validate YAML defaults; malformed settings must never silently change a run."""
    actions = {action.dest: action for action in parser._actions}
    defaults = {}
    for key, value in flat_config.items():
        if key in MODEL_KEYS:
            continue
        if key not in YAML_KEY_TO_DEST:
            raise ValueError(f"Unknown configuration key: {key}")
        dest = YAML_KEY_TO_DEST[key]
        action = actions[dest]
        if value is None:
            if dest not in {"resume", "pretrained", "pin_memory", "output_dir", "run_id"}:
                raise ValueError(f"{key} cannot be null")
            defaults[dest] = None
            continue
        if isinstance(action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction)):
            if isinstance(value, str):
                choices = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}
                if value.lower() not in choices:
                    raise ValueError(f"{key} requires a boolean")
                value = choices[value.lower()]
            if not isinstance(value, bool):
                raise ValueError(f"{key} requires a boolean")
        elif action.type is not None:
            if isinstance(value, bool) or (action.type is int and isinstance(value, float) and not value.is_integer()):
                raise ValueError(f"Invalid value for {key}: {value!r}")
            try:
                value = action.type(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid value for {key}: {value!r}") from exc
        if action.choices is not None and value not in action.choices:
            raise ValueError(f"Invalid {key}: {value!r}; expected {action.choices}")
        defaults[dest] = value
    return defaults


def generate_run_dir(
    run_root: Path | str,
    model_name: str,
    seed: int | str | None = None,
    timestamp: str | None = None,
) -> Path:
    """Generate collision-defended timestamped run directory in <model>_seed<seed>_<YYYY-MM-DD_HHhMM> format."""
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    clean_model_name = str(model_name).replace(" ", "-").replace("/", "-")
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%M")

    if seed is not None:
        seed_str = str(seed).strip()
        seed_part = f"_{seed_str}" if seed_str.lower().startswith("seed") else f"_seed{seed_str}"
    else:
        seed_part = ""

    base_name = f"{clean_model_name}{seed_part}_{timestamp}"
    candidate = run_root / base_name
    try:
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    except FileExistsError:
        if not any(candidate.iterdir()):
            return candidate

    for counter in range(1, 100):
        candidate = run_root / f"{base_name}_{counter:02d}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            if not any(candidate.iterdir()):
                return candidate

    micro = datetime.now().strftime("%f")
    candidate = run_root / f"{base_name}_{micro}"
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate



def load_merged_config(config_path: str | Path | None = None, base_path: str | Path | None = None) -> dict[str, Any]:
    """Load base and model YAML configs and recursively merge them."""
    if base_path is None:
        base_path = Path(__file__).resolve().parent / "base.yaml"
    else:
        base_path = Path(base_path)

    base_cfg: dict[str, Any] = {}
    if not base_path.is_file():
        raise FileNotFoundError(f"Base configuration not found: {base_path}")
    if base_path.is_file():
        with base_path.open("r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
    if not isinstance(base_cfg, dict):
        raise ValueError("Base YAML must be a mapping")

    if config_path is None:
        return base_cfg

    cfg_path = Path(config_path)
    if not cfg_path.is_file() and cfg_path.parent == Path("."):
        # Check relative to models/ directory
        alt_path = Path(__file__).resolve().parent / "models" / f"{cfg_path.name}"
        if alt_path.is_file():
            cfg_path = alt_path
        elif not cfg_path.name.endswith(".yaml"):
            alt_path_yaml = Path(__file__).resolve().parent / "models" / f"{cfg_path.name}.yaml"
            if alt_path_yaml.is_file():
                cfg_path = alt_path_yaml

    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}
    if not isinstance(model_cfg, dict):
        raise ValueError("Model YAML must be a mapping")

    return deep_merge(base_cfg, model_cfg)
