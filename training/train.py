"""Train the prompt-free M2/M2-Plus/M3 ablations (hierarchical YAML configuration & CLI)."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.config.config_utils import (
    coerce_config_to_parser_types,
    flatten_config,
    generate_run_dir,
    load_merged_config,
)
from training.models.cmspa_net import CONFIGS, CMSPANet
from training.models import model_from_config
from training.run_layout import RunLayout
from training.trainer.trainer import Trainer, seed_everything


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="training/config/models/cmspa_net.yaml",
        help="Path to model YAML config, or legacy config name: R50-ViT-B_16, testing",
    )
    parser.add_argument(
        "--base-config",
        default=str(PROJECT_ROOT / "training" / "config" / "base.yaml"),
        help="Path to base YAML config",
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        default="E:/STUDY/DATASET/MyoPS380/Processed_data",
    )
    parser.add_argument(
        "--list-dir",
        "--list_dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "splits"),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; defaults to outputs/runs/{model_name}_seed{seed}_{YYYY-MM-DD_HHhMM}",
    )
    parser.add_argument(
        "--run-root",
        default=str(PROJECT_ROOT / "outputs" / "runs"),
        help="Root directory for automated timestamped runs",
    )
    parser.add_argument("--run-id", default=None, help="Named run directory under --run-root")
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument(
        "--aar-weight",
        type=float,
        default=0.0,
        help="Weight for conditional Area-at-Risk (AAR / Edema-Inclusive) auxiliary Dice loss (default: 0.0)",
    )
    parser.add_argument(
        "--scar-weight",
        type=float,
        default=0.0,
        help="Weight for conditional Scar auxiliary Dice loss (default: 0.0)",
    )
    parser.add_argument(
        "--wall-weight",
        type=float,
        default=0.0,
        help="Weight for conditional myocardial wall auxiliary loss (default: 0.0)",
    )
    parser.add_argument(
        "--inclusion-weight",
        type=float,
        default=0.0,
        help="Weight for hierarchical pathology inclusion loss (Scar in AAR) (default: 0.0)",
    )
    parser.add_argument(
        "--dice-class-weights",
        nargs="+",
        type=float,
        default=None,
        help="Class weights for Dice loss, e.g. 0.0 1.0 2.0 2.0 (default: equal weights)",
    )
    parser.add_argument(
        "--ce-class-weights",
        nargs="+",
        type=float,
        default=None,
        help="Class weights for Cross-Entropy loss, e.g. 0.2 1.0 2.0 2.0 (default: equal weights)",
    )
    parser.add_argument("--sampler", choices=("none", "rare"), default="none")
    parser.add_argument("--rare-boost", type=float, default=2.0)
    parser.add_argument("--foreground-boost", type=float, default=1.3)
    parser.add_argument("--ablation", choices=["M2", "M2-Plus", "M2-PLUS", "M2-PRO", "M2PRO", "M2-MAX", "M2MAX", "M2-MAX-PRO", "M2MAXPRO", "M2-MAX-V2", "M2MAXV2", "M2-MAX-V4", "M2MAXV4", "M2-MAX-V5", "M2MAXV5", "M2-MAX-V6", "M2MAXV6", "M2-MAX-V7", "M2MAXV7", "M3", "H-CMSPA", "HCMSPA"], default="M3")
    parser.add_argument("--epochs", "--max_epochs", dest="max_epochs", type=int, default=300)
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=16,
        help="microbatch on one device",
    )
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--lr", "--base_lr", dest="base_lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=128)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="recompute encoder blocks to reduce activation memory",
    )
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="held-out TRAIN patients if val.txt is absent",
    )
    parser.add_argument("--label-order", choices=["legacy", "canonical"], default="legacy")
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="0 disables early stopping; measured in validation epochs",
    )
    parser.add_argument("--min-delta", type=float, default=0.0, help="minimum Dice gain to reset patience")
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="optional archival checkpoint cadence; best/last always saved",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume",
        default=None,
        help="complete state from this pipeline; keep original training settings",
    )
    parser.add_argument("--pretrained", default=None, help="explicit R50-ViT-B_16.npz encoder initialization")
    parser.add_argument("--init-weights", default=None, help="checkpoint path to initialize full model weights before training")
    parser.add_argument(
        "--epochs-per-run",
        type=int,
        default=0,
        help="pause after N additional epochs while preserving the planned LR schedule",
    )
    return parser


def parse_args(argv=None):
    """Resolve base YAML < model YAML < explicit CLI, without starting training."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="training/config/models/cmspa_net.yaml")
    pre_parser.add_argument("--base-config", default=str(PROJECT_ROOT / "training/config/base.yaml"))
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_name = pre_args.config
    if config_name in CONFIGS:
        config_name = "testing.yaml" if config_name == "testing" else "cmspa_net.yaml"
    config_path = Path(config_name)
    if not config_path.is_absolute() and (PROJECT_ROOT / config_path).is_file():
        config_path = PROJECT_ROOT / config_path
    merged = load_merged_config(config_path, pre_args.base_config)
    parser = build_parser()
    parser.set_defaults(**coerce_config_to_parser_types(flatten_config(merged), parser))
    args = parser.parse_args(argv)
    for key in ("max_epochs", "batch_size", "accum_steps", "img_size", "cpu_threads", "log_every"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if args.img_size < 32 or args.img_size % 16:
        parser.error("img_size must be a multiple of 16 and >=32")
    if args.base_lr <= 0 or args.weight_decay < 0 or not 0 < args.val_fraction < 1:
        parser.error("Require lr>0, weight_decay>=0, 0<val_fraction<1")
    for key in ("base_lr", "weight_decay", "val_fraction", "min_delta", "clip_grad", "ce_weight", "dice_weight", "aar_weight", "scar_weight"):
        if not math.isfinite(getattr(args, key)):
            parser.error(f"{key} must be finite")
    for key in ("num_workers", "patience", "min_delta", "save_every", "epochs_per_run", "clip_grad", "ce_weight", "dice_weight", "aar_weight", "scar_weight"):
        if getattr(args, key) < 0:
            parser.error(f"{key} must be nonnegative")
    if args.ce_weight + args.dice_weight <= 0:
        parser.error("At least one loss weight must be positive")
    if any(not math.isfinite(v) or v <= 0 for v in (args.rare_boost, args.foreground_boost)):
        parser.error("Sampler weights must be finite and positive")
    if args.resume and args.pretrained:
        parser.error("--resume and --pretrained are mutually exclusive")
    if args.run_id and (args.run_id in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in args.run_id)):
        parser.error("run-id may contain only letters, digits, dots, underscores, and hyphens")
    if args.run_id and args.output_dir not in (None, "auto"):
        parser.error("Use either --run-id or --output-dir")
    if merged["model"].get("num_classes", 4) != 4:
        parser.error("This data contract requires four canonical classes")
    return args, merged


def main(argv=None):
    args, merged = parse_args(argv)
    for key in ("data_root", "list_dir"):
        path = Path(getattr(args, key)).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"{key} directory does not exist: {path}")
        setattr(args, key, str(path))
    if args.output_dir in (None, "auto"):
        if args.run_id:
            output = Path(args.run_root) / args.run_id
        elif args.resume:
            output = RunLayout.from_checkpoint(args.resume).root
        else:
            output = generate_run_dir(
                args.run_root,
                merged["model"].get("model_name", "CMSPA-Net"),
                seed=args.seed,
            )
    else:
        output = Path(args.output_dir)
    output = output.resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("Run directory is not empty; choose a new run-id or use --resume")
    if args.resume:
        if RunLayout.from_checkpoint(args.resume).root != output or Path(args.resume).name != "last.pth":
            raise ValueError("Resume from last.pth into its original run directory")
    args.output_dir = str(output)
    args.model_name = merged["model"].get("model_name", "CMSPA-Net")
    section = merged["model"]
    config = copy.deepcopy(CONFIGS["R50-ViT-B_16"])
    config.resnet.num_layers = tuple(section["resnet"]["num_layers"])
    config.resnet.width_factor = float(section["resnet"]["width_factor"])
    config.transformer.dropout_rate = float(section.get("transformer", {}).get("dropout_rate", 0.1))
    for key in ("decoder_channels", "skip_channels", "fused_channels", "n_skip", "cross_attention_heads", "classifier", "activation"):
        if key in section:
            config[key] = tuple(section[key]) if key == "decoder_channels" else section[key]
    config.ablation = args.ablation
    config.gradient_checkpointing = args.gradient_checkpointing
    config.n_classes = 4
    if "architecture" in section:
        config.architecture = section["architecture"]
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed, args.deterministic)
    model = model_from_config(config, img_size=args.img_size, num_classes=4, ablation=args.ablation)
    if args.pretrained:
        model.load_pretrained_encoders(args.pretrained)
    elif getattr(args, "init_weights", None):
        ckpt = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        sd = ckpt["model"]
        if hasattr(model.segmentation_head, "base_head") and "segmentation_head.0.weight" in sd:
            sd["segmentation_head.base_head.weight"] = sd.pop("segmentation_head.0.weight")
            sd["segmentation_head.base_head.bias"] = sd.pop("segmentation_head.0.bias")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Successfully loaded initial model weights from {args.init_weights} (missing: {len(missing)}, unexpected: {len(unexpected)})")
    return Trainer(model, args, output).fit()


if __name__ == "__main__":
    main()
