"""Prepare or validate MyoPS cache, train M2/M3, then evaluate held-out volumes."""
from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import subprocess
import sys

import yaml

from training.config.config_utils import load_merged_config

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("scar.pipeline")


def run_command(command: list[str], description: str) -> None:
    LOGGER.info("%s", description)
    subprocess.run(command, cwd=ROOT, check=True)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/config/models/cmspa_net.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--data-root", help="Processed cache, containing bSSFP/LGE/T2w")
    parser.add_argument("--raw-root", help="Aligned raw NIfTI root, used only to create a new cache")
    parser.add_argument("--list-dir", help="Patient manifests; defaults to training YAML")
    parser.add_argument("--test-list", default="preprocessing/splits/test_vol.txt")
    parser.add_argument("--skip-cache", action="store_true", help="Require and reuse an existing cache")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--eval-split", choices=["test_vol", "val_vol"], default="test_vol")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--run-root", help="Experiment directory, defaults to training YAML")
    parser.add_argument("--label-order", choices=["legacy", "canonical"])
    parser.add_argument("--normalization", choices=["unit255", "unit", "percentile"])
    for name, kind in (("epochs", int), ("batch-size", int), ("accum-steps", int),
                       ("num-workers", int), ("cpu-threads", int), ("lr", float)):
        parser.add_argument(f"--{name}", type=kind)
    parser.add_argument("--device")
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"])
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed; defaults to config YAML")
    return parser


def main(argv=None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config = load_merged_config(project_path(args.config))
    with (ROOT / "preprocessing/config.yaml").open(encoding="utf-8") as stream:
        prep_config = yaml.safe_load(stream)
    data_root = project_path(args.data_root or config["data"]["data_root"])
    list_dir = project_path(args.list_dir or config["data"]["list_dir"])
    raw_root = project_path(args.raw_root or prep_config["data"]["raw_root"])
    test_list = project_path(args.test_list)
    run_root = project_path(args.run_root or config["outputs"]["run_root"])
    seed = args.seed if args.seed is not None else config.get("seed", 1234)
    model_name = config["model"].get("model_name") or config["model"].get("architecture", config["model"]["ablation"])
    clean_model = str(model_name).replace(" ", "-").replace("/", "-")
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    run_id = args.run_id or f"{clean_model}_seed{seed}_{timestamp}"
    if Path(run_id).name != run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        parser.error("--run-id must be a single directory name")
    run_dir = run_root / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"Run directory is not empty: {run_dir}; use training/train.py --resume to resume")
    python = sys.executable
    cache_present = all((data_root / modality / "train_npz").is_dir()
                        for modality in ("bSSFP", "LGE", "T2w"))
    if args.skip_cache and not cache_present:
        parser.error(f"--skip-cache requires an existing three-modality cache: {data_root}")
    if cache_present:
        run_command([python, "preprocessing/build_splits.py", "--data-root", str(data_root),
                     "--list-dir", str(list_dir), "--test-list", str(test_list),
                     "--seed", str(seed), "--val-fraction", str(config["data"]["val_fraction"])],
                    "1/4: Validate or create patient manifests for the existing cache")
    else:
        normalization = args.normalization or prep_config["normalization"]
        raw_label_order = args.label_order or prep_config["label_order"]
        if raw_label_order == "auto":
            parser.error("Creating a cache requires an explicit raw --label-order legacy or canonical")
        run_command([python, "preprocessing/process_and_save.py", "--src-path", str(raw_root),
                     "--dst-path", str(data_root), "--list-dir", str(list_dir),
                     "--test-list", str(test_list), "--label-order", raw_label_order,
                     "--normalization", normalization, "--seed", str(seed),
                     "--val-fraction", str(config["data"]["val_fraction"])],
                    "1/4: Package aligned NIfTI into a new cache and patient manifests")
    # Newly packaged files are canonical regardless of the raw encoding.
    label_order = (args.label_order or config["data"]["label_order"]) if cache_present else "canonical"
    run_command([python, "preprocessing/verify.py", "--data-root", str(data_root),
                 "--list-dir", str(list_dir), "--label-order", label_order],
                "2/4: Check synchronized modalities, labels, splits and geometry")
    train_command = [python, "training/train.py", "--config", str(project_path(args.config)),
                     "--run-id", run_id, "--run-root", str(run_root), "--data-root", str(data_root),
                     "--list-dir", str(list_dir), "--label-order", label_order,
                     "--seed", str(seed)]
    for name in ("epochs", "batch_size", "accum_steps", "num_workers", "cpu_threads", "lr", "device", "amp"):
        value = getattr(args, name)
        if value is not None:
            train_command.extend(["--" + name.replace("_", "-"), str(value)])
    for name in ("tensorboard", "gradient_checkpointing"):
        value = getattr(args, name)
        if value is not None:
            train_command.append("--" + ("" if value else "no-") + name.replace("_", "-"))
    run_command(train_command, "3/4: Train with validation-based checkpoint selection")
    checkpoint = run_dir / "checkpoints" / "best.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training did not produce the best checkpoint: {checkpoint}")
    if not args.skip_evaluate:
        evaluation = [python, "training/evaluate.py", "--checkpoint", str(checkpoint),
                      "--data-root", str(data_root), "--split", args.eval_split,
                      "--batch-size", str(args.eval_batch_size), "--label-order", label_order]
        for name in ("device", "amp", "cpu_threads"):
            value = getattr(args, name)
            if value is not None:
                evaluation.extend(["--" + name.replace("_", "-"), str(value)])
        run_command(evaluation, "4/4: Evaluate complete held-out volumes")
    LOGGER.info("Completed: %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
