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
    from training.dataset.benchmark_profiles import DATASET_IDS
    parser.add_argument("--dataset", choices=DATASET_IDS, default=None,
                        help="CLI > YAML data.dataset_id > myops380")
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
    model_config_path = project_path(args.config)
    config_name = str(model_config_path) if model_config_path.is_file() else args.config
    config = load_merged_config(config_name, dataset_id=args.dataset)
    args.dataset = config["data"]["dataset_id"]
    with (ROOT / "preprocessing/config.yaml").open(encoding="utf-8") as stream:
        prep_config = yaml.safe_load(stream)
    data_root = project_path(args.data_root or config["data"]["data_root"])
    list_default = (data_root / "lists" if args.dataset != "myops380" and args.data_root is not None
                    else config["data"]["list_dir"])
    list_dir = project_path(args.list_dir or list_default)
    raw_default = "E:/STUDY/DATASET/Myo_train" if args.dataset == "myopspp_bc80" else prep_config["data"]["raw_root"]
    from training.dataset.benchmark_profiles import ROI_PROFILES
    if args.dataset in ROI_PROFILES:
        raw_default = ROOT.parent / "MyoPSpp_preprocessed_2026-10-03" / ROI_PROFILES[args.dataset]["export_id"]
    raw_root = project_path(args.raw_root or raw_default)
    test_list = project_path(args.test_list)
    run_root = project_path(args.run_root or config["outputs"]["run_root"])
    seed = args.seed if args.seed is not None else config.get("seed", 1234)
    model_name = config["model"].get("model_name") or config["model"].get("architecture", config["model"]["ablation"])
    clean_model = str(model_name).replace(" ", "-").replace("/", "-")
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    run_id = args.run_id or config["outputs"].get("run_id") or f"{clean_model}_seed{seed}_{timestamp}"
    if Path(run_id).name != run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        parser.error("--run-id must be a single directory name")
    configured_output = None
    # Match train's YAML alias resolution: the final mapped output-dir wins.
    for key, value in config["outputs"].items():
        if key in ("dir", "output_dir"):
            configured_output = value
    use_output_dir = args.run_id is None and configured_output not in (None, "auto")
    if use_output_dir and config["outputs"].get("run_id") is not None:
        parser.error("Use either outputs.dir or outputs.run_id in YAML")
    run_dir = project_path(configured_output) if use_output_dir else run_root / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"Run directory is not empty: {run_dir}; use training/train.py --resume to resume")
    python = sys.executable
    cache_present = all((data_root / modality / "train_npz").is_dir()
                        for modality in ("bSSFP", "LGE", "T2w"))
    if args.skip_cache and not cache_present:
        parser.error(f"--skip-cache requires an existing three-modality cache: {data_root}")
    if args.dataset in ROI_PROFILES:
        if args.label_order not in (None, "canonical") or args.normalization is not None:
            parser.error("ROI exports require canonical cache labels and preserve their existing normalization; omit --normalization")
        if list_dir != data_root / "lists" and not cache_present:
            parser.error("New ROI caches package manifests at --data-root/lists")
        if not cache_present:
            run_command([python, "-m", "preprocessing.myopspp_roi", "--dataset", args.dataset,
                         "--src-path", str(raw_root), "--dst-path", str(data_root)], "1/4: Adapt exported cardiac ROIs")
    elif args.dataset == "myopspp_bc80":
        if args.label_order not in (None, "canonical") or args.normalization not in (None, "percentile"):
            parser.error("myopspp_bc80 requires canonical cache labels and percentile normalization")
        if list_dir != data_root / "lists" and not cache_present:
            parser.error("New MyoPS++ caches package manifests at --data-root/lists")
        if not cache_present:
            run_command([python, "-m", "preprocessing.myopspp", "--src-path", str(raw_root),
                         "--dst-path", str(data_root)], "1/4: Package fixed MyoPS++ B/C benchmark")
    elif cache_present:
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
                 "--list-dir", str(list_dir), "--label-order", label_order, "--dataset", args.dataset],
                "2/4: Check synchronized modalities, labels, splits and geometry")
    train_command = [python, "training/train.py", "--config", config_name,
                     "--run-root", str(run_root), "--data-root", str(data_root),
                     "--list-dir", str(list_dir), "--label-order", label_order,
                     "--seed", str(seed), "--dataset", args.dataset]
    if use_output_dir:
        train_command.extend(["--output-dir", str(run_dir)])
    else:
        train_command.extend(["--run-id", run_id, "--output-dir", "auto"])
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
