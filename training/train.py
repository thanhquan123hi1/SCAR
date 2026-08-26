"""Entry point: train a multi-view LGE segmentation model with reproducible seed determinism.

Usage:
    python training/train.py \
        --config training/config/models/unet_3d.yaml \
        --run-id unet3d_lge_sax_exp01
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Make repo root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.dataset.lge_dataset import LgeLaxDataset, LgeSaxDataset
from training.loss import build_loss
from training.models import build_model
from training.trainer.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Ensure 100% deterministic reproducibility for scientific experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def merge_configs(base_path: Path, model_path: Path) -> dict:
    """Load base config then override with model-specific config."""
    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    model_cfg = yaml.safe_load(model_path.read_text(encoding="utf-8"))

    def deep_merge(base_d: dict, override_d: dict) -> dict:
        result = dict(base_d)
        for key, value in override_d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    return deep_merge(base, model_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LGE segmentation model.")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--base-config", default="training/config/base.yaml")
    args = parser.parse_args()

    config = merge_configs(ROOT / args.base_config, ROOT / args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    logger.info("Deterministic seed set to: %d", seed)

    # Device
    req_device = str(config.get("device", "auto")).lower()
    if req_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(req_device)
    logger.info("Using device: %s", device)

    # Run directory
    run_dir = ROOT / config.get("outputs", {}).get("run_root", "outputs/runs") / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot config
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.dump(config, allow_unicode=True), encoding="utf-8"
    )

    # Load splits
    splits_dir = ROOT / config["data"]["splits_dir"]
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "validation.csv")

    view = str(config.get("view", "SAX")).upper()
    train_df = train_df[train_df["view"] == view]
    val_df = val_df[val_df["view"] == view]
    logger.info("Dataset filtered: Train=%d, Val=%d (View=%s)", len(train_df), len(val_df), view)

    # Preprocessing params
    pp = config.get("preprocessing", {})
    target_shape = tuple(pp["target_shape"])
    target_spacing = tuple(pp["target_spacing"])
    percentiles = tuple(pp["intensity_percentiles"]) if pp.get("intensity_percentiles") else None

    # Cache dir (optional fast loading)
    cache_dir = ROOT / config["data"].get("cache_dir", "data/processed/cache")
    raw_root = ROOT / config["data"]["raw_root"]

    DatasetClass = LgeSaxDataset if view == "SAX" else LgeLaxDataset

    train_ds = DatasetClass(
        records=train_df,
        data_root=raw_root,
        cache_dir=cache_dir if cache_dir.exists() else None,
        target_shape=target_shape,
        target_spacing=target_spacing,
        intensity_percentiles=percentiles,
        augment=bool(config.get("training", {}).get("augment", True)),
    )
    val_ds = DatasetClass(
        records=val_df,
        data_root=raw_root,
        cache_dir=cache_dir if cache_dir.exists() else None,
        target_shape=target_shape,
        target_spacing=target_spacing,
        intensity_percentiles=percentiles,
        augment=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=bool(config["data"].get("pin_memory", False)),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
    )

    # Model
    num_classes = int(config.get("num_classes", 5))
    model = build_model(
        config["model_name"],
        **{**config.get("model", {}), "num_classes": num_classes},
    ).to(device)

    # Optimizer & Scheduler
    lr = float(config["training"]["learning_rate"])
    wd = float(config["training"].get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(config["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Loss
    loss_cfg = config.get("loss", {})
    loss_fn = build_loss(
        loss_cfg.get("name", "ce_dice"),
        num_classes=num_classes,
        ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        run_dir=run_dir,
        config=config,
        num_classes=num_classes,
        view=view,
    )

    logger.info("Starting training run '%s' for %d epochs...", args.run_id, epochs)
    trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        early_stopping_cfg=config["training"].get("early_stopping"),
    )
    logger.info("Training complete. Artifacts saved to %s", run_dir)


if __name__ == "__main__":
    main()
