"""Observable single-device training, validation, and resumable checkpoints."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.dataset.data_contract import CLASS_NAMES, ensure_patient_splits, lock_benchmark_data
from training.dataset.myops_dataset import (
    MyopsDataset,
    RandomGenerator,
    ResizeGenerator,
)
from training.loss.losses import LOSS_PROTOCOL_VERSION, SegmentationLoss
from training.metrics.confusion_meter import ConfusionMeter
from training.dataset.sampler import build_rare_class_sampler
from training.predict import predict_volume
from training.run_layout import RunLayout
from training.metrics.surface_distance import protocol_for_dataset, dataset_rows, summarize_dataset_rows


def validate_volumes(model, dataset, img_size, batch_size, device, amp_dtype, dataset_id="myops380"):
    """Selection uses native-grid full patients, never pixel-pooled slice scores."""
    rows = []
    for sample in dataset:
        if dataset_id == "myopspp_bc80" and not sample["has_geometry"]:
            raise ValueError("MyoPS++ validation requires native mm geometry")
        prediction = predict_volume(model, [sample[k] for k in ("image", "image1", "image2")],
                                    img_size, batch_size, device, amp_dtype)
        rows.extend(dataset_rows(prediction, sample["label"], sample["case_name"], compute_distance=False,
                                 dataset_id=dataset_id, spacing=sample.get("spacing")))
    return summarize_dataset_rows(rows, dataset_id=dataset_id)


def inclusive_validation_score(metrics):
    values = [metrics.get(region, {}).get("mean_dice") for region in ("scar", "edema_inclusive")]
    if any(value is None for value in values):
        return None
    score = float(np.mean(values))
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid inclusive validation Dice")
    return score


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id: int):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(name: str = "auto") -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This pipeline supports CPU or CUDA.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch installation.")
    return device


def resolve_amp(mode: str, device: torch.device):
    if mode == "none" or (mode == "auto" and device.type == "cpu"):
        return None
    if device.type != "cuda":
        raise ValueError("Mixed precision is enabled only on CUDA; use --amp none on CPU.")
    if mode not in {"auto", "none", "fp16", "bf16"}:
        raise ValueError(f"Unknown AMP mode: {mode}")
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bfloat16.")
        return torch.bfloat16
    return torch.float16


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path | str, payload: Any):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_checkpoint(path: Path | str, payload: dict[str, Any]):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError(
            "Expected a SCAR M0-M3 pipeline checkpoint (format_version=1)."
        )
    if tuple(checkpoint.get("class_names", ())) != CLASS_NAMES:
        raise ValueError("Checkpoint label semantics do not match this pipeline.")
    return checkpoint


def rng_state() -> dict[str, Any]:
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]):
    random.setstate(state["python"])
    s = state["numpy"]
    np.random.set_state((s[0], np.asarray(s[1], dtype=np.uint32), s[2], s[3], s[4]))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def make_logger(directory: Path) -> logging.Logger:
    logger = logging.getLogger("scar.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(directory / "train.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def append_metrics_csv(path, record):
    """Expand old CSV schemas on resume; historical new metrics stay blank."""
    path = Path(path)
    fields = list(record)
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            old_fields = reader.fieldnames
            fields = old_fields + [key for key in record if key not in old_fields]
            if fields != old_fields:
                temporary = path.with_suffix(".csv.tmp")
                with temporary.open("w", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(output, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(reader)
        if fields != old_fields:
            temporary.replace(path)
    new = not path.exists() or not path.stat().st_size
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(record)


def _validate_resume_loss_config(saved_args: dict, args, saved_version: int | None = None) -> None:
    """Missing auxiliary settings in older checkpoints mean disabled/default."""
    for key in ("aar_weight", "scar_weight", "wall_weight", "inclusion_weight",
                "dice_class_weights", "ce_class_weights"):
        default = None if key.endswith("class_weights") else 0.0
        previous, current = saved_args.get(key, default), getattr(args, key, default)
        if previous != current:
            raise ValueError(f"Resume changes {key}: {previous} -> {current}. Use the original training settings.")
    if saved_version is not None and saved_version != LOSS_PROTOCOL_VERSION:
        if getattr(args, "inclusion_weight", 0.0) > 0 or getattr(args, "ce_class_weights", None) is not None:
            raise ValueError("Resume loss protocol differs: weighted CE/inclusion semantics were corrected. "
                             "Start a new run with --init-weights to use the updated objective.")


def _configs_equal(c1: Any, c2: Any) -> bool:
    """Recursively compare configurations normalizing sequences (tuples and lists)."""
    if isinstance(c1, dict) and isinstance(c2, dict):
        if set(c1.keys()) != set(c2.keys()):
            return False
        return all(_configs_equal(c1[k], c2[k]) for k in c1)
    if isinstance(c1, (list, tuple)) and isinstance(c2, (list, tuple)):
        if len(c1) != len(c2):
            return False
        return all(_configs_equal(v1, v2) for v1, v2 in zip(c1, c2))
    return c1 == c2


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: Any = None,
    accum_steps: int = 1,
    clip_grad: float = 1.0,
    log_every: int = 20,
    logger: logging.Logger | None = None,
    writer: Any = None,
    global_step: int = 0,
) -> tuple[dict[str, Any], int]:
    training = optimizer is not None
    if accum_steps < 1 or log_every < 1:
        raise ValueError("accum_steps and log_every must be positive")
    if training and scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    model.train(training)
    meter = ConfusionMeter(device=device)
    sums = torch.zeros(3, device=device, dtype=torch.float64)
    extra_sums = {}
    sample_count, skipped_steps, grad_norm_sum, updates, group_samples = 0, 0, 0.0, 0, 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader):
            images = [batch[key].to(device, non_blocking=loader.pin_memory) for key in ("image", "image1", "image2")]
            target = batch["label"].to(device, non_blocking=loader.pin_memory)
            batch_count = target.shape[0]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                output = getattr(model, "forward_for_loss", model)(*images)
                losses = criterion(output, target)
                logits = output["logits"] if isinstance(output, dict) else output
            if not torch.stack([torch.isfinite(v) for v in losses.values()]).all():
                raise FloatingPointError(f"Non-finite loss at batch {batch_index}: {batch.get('case_name')}")
            sums += torch.stack([losses[key].detach() for key in ("loss", "ce", "dice_loss")]).double() * batch_count
            for key, value in losses.items():
                if key not in ("loss", "ce", "dice_loss"):
                    extra_sums[key] = extra_sums.get(key, 0) + value.detach().double() * batch_count
            sample_count += batch_count
            meter.update(logits.detach().argmax(1), target)
            if training:
                scaler.scale(losses["loss"] * batch_count).backward()
                group_samples += batch_count
                boundary = (batch_index + 1) % accum_steps == 0 or batch_index + 1 == len(loader)
                if boundary:
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(group_samples)
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_grad if clip_grad > 0 else float("inf")
                    )
                    finite_gradient = bool(torch.isfinite(norm))
                    if not finite_gradient and not scaler.is_enabled():
                        raise FloatingPointError(f"Non-finite gradient at batch {batch_index}.")
                    if not finite_gradient and scaler.is_enabled():
                        finite_entries = all(
                            bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
                        )
                        if finite_entries:
                            raise FloatingPointError(
                                "Gradient norm overflowed despite finite entries; reduce the loss/gradient scale."
                            )
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped = scaler.get_scale() < old_scale
                    if skipped:
                        skipped_steps += 1
                        if logger:
                            logger.warning(
                                "AMP overflow: optimizer update skipped at batch %d, new scale %.1f",
                                batch_index,
                                scaler.get_scale(),
                            )
                    else:
                        if scheduler is not None:
                            scheduler.step()
                        global_step += 1
                        updates += 1
                        grad_norm_sum += float(norm)
                    if writer and (global_step % log_every == 0 or skipped):
                        writer.add_scalar("step/loss", float(losses["loss"].detach()), global_step)
                        writer.add_scalar("step/grad_norm", float(norm), global_step)
                        writer.add_scalar("step/amp_scale", scaler.get_scale(), global_step)
                    optimizer.zero_grad(set_to_none=True)
                    group_samples = 0
            if logger and ((batch_index + 1) % log_every == 0 or batch_index + 1 == len(loader)):
                logger.info(
                    "%s batch %d/%d | loss %.5f | ce %.5f | dice_loss %.5f",
                    "train" if training else "val",
                    batch_index + 1,
                    len(loader),
                    *[float(v) / sample_count for v in sums],
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if sample_count == 0:
        raise ValueError("Cannot run an epoch with an empty data loader")
    metrics = {key: float(value) / sample_count for key, value in zip(("loss", "ce", "dice_loss"), sums)}
    metrics.update(meter.compute())
    metrics.update({key: float(value) / sample_count for key, value in extra_sums.items()})
    metrics.update(
        seconds=elapsed,
        samples_per_second=sample_count / elapsed,
        samples=sample_count,
        gpu_peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        gpu_peak_reserved_mb=torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0,
    )
    if training:
        metrics.update(
            grad_norm=grad_norm_sum / max(updates, 1),
            amp_skipped_steps=skipped_steps,
            optimizer_updates=updates,
            amp_scale=scaler.get_scale(),
        )
        if updates == 0:
            raise FloatingPointError("No optimizer updates succeeded in this epoch. Check AMP scale/input ranges.")
    return metrics, global_step


class EarlyStopping:
    """Early stopping monitor based on validation score."""

    def __init__(self, patience: int = 0, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.early_stop = False

    def step(self, score: float) -> bool:
        if self.patience <= 0:
            return False
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return improved


class Trainer:
    """Run the configured dataset, optimizer, validation, and checkpoint pipeline.

    Training configuration is the single source of truth. ``fit`` never silently
    replaces caller-supplied optimizers or data loaders.
    """

    def __init__(self, model: nn.Module, args: Any, run_dir: Path | str):
        self.model = model
        self.args = args
        self.run_dir = Path(run_dir)

    def fit(self) -> dict[str, Any]:
        return trainer_Myops(self.args, self.model, self.run_dir)


def trainer_Myops(args, model, snapshot_path):
    dataset_id = getattr(args, "dataset_id", "myops380")
    benchmark_protocol = protocol_for_dataset(dataset_id)
    resume_checkpoint = load_checkpoint(args.resume) if args.resume else None
    if resume_checkpoint is not None and resume_checkpoint.get("benchmark_protocol") != benchmark_protocol:
        raise ValueError("Cannot resume a checkpoint from a different dataset/metric protocol")
    directory = Path(snapshot_path)
    directory.mkdir(parents=True, exist_ok=True)
    layout = (RunLayout.from_checkpoint(args.resume) if args.resume else
              RunLayout.create(directory, args.model_name, args.seed))
    logger = make_logger(layout.logs)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp(args.amp, device)
    split_dir = ensure_patient_splits(
        args.list_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        output_dir=layout.logs / "splits",
    )
    split_hashes = {
        name: hashlib.sha256((split_dir / f"{name}.txt").read_bytes()).hexdigest()
        for name in ("train", "val", "val_vol", "test_vol")
        if (split_dir / f"{name}.txt").is_file()
    }
    benchmark_data = lock_benchmark_data(args.data_root, split_dir, args.label_order, dataset_id=dataset_id)
    roots = [str(Path(args.data_root) / modality / "train_npz") for modality in ("bSSFP", "LGE", "T2w")]
    datasets = [
        MyopsDataset(
            *roots,
            str(split_dir),
            split,
            transform=(RandomGenerator if split == "train" else ResizeGenerator)([args.img_size, args.img_size]),
            label_order=args.label_order,
        )
        for split in ("train", "val")
    ]
    if any(len(dataset) == 0 for dataset in datasets):
        raise ValueError("Training and validation must both be nonempty.")
    pin = args.pin_memory if args.pin_memory is not None else device.type == "cuda"
    generator = torch.Generator().manual_seed(args.seed)
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin,
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )
    sampler = (build_rare_class_sampler(datasets[0], rare_boost=args.rare_boost,
                                       foreground_boost=args.foreground_boost, generator=generator)
               if args.sampler == "rare" else None)
    trainloader = DataLoader(datasets[0], shuffle=sampler is None, sampler=sampler,
                             generator=generator, **common)
    valloader = DataLoader(datasets[1], shuffle=False, generator=torch.Generator().manual_seed(args.seed + 1), **common)
    validation_volumes = MyopsDataset(
        *[Path(args.data_root) / modality / "val_vol_h5" for modality in ("bSSFP", "LGE", "T2w")],
        split_dir, "val_vol", label_order=args.label_order,
    )
    model.to(device)
    criterion = SegmentationLoss(
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        aar_weight=getattr(args, "aar_weight", 0.0),
        scar_weight=getattr(args, "scar_weight", 0.0),
        wall_weight=getattr(args, "wall_weight", 0.0),
        inclusion_weight=getattr(args, "inclusion_weight", 0.0),
        dice_class_weights=getattr(args, "dice_class_weights", None),
        ce_class_weights=getattr(args, "ce_class_weights", None),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, foreach=False)
    total_updates = args.max_epochs * math.ceil(len(trainloader) / args.accum_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1.0 - step / total_updates) ** 0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    start_epoch, global_step, best_score, best_epoch, bad_epochs = 0, 0, -float("inf"), -1, 0
    patience_score = -float("inf")
    best_inclusive_score = -float("inf")
    model_config = model.config.to_dict()
    metadata_path = Path(args.data_root) / "dataset_metadata.json"
    data_provenance = {}
    if metadata_path.is_file():
        data_provenance = {"metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
                           "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()}

    if args.resume:
        checkpoint = resume_checkpoint
        if checkpoint.get("benchmark_data") != benchmark_data:
            raise ValueError("Cache bytes or source label convention changed since training")
        if Path(checkpoint["args"]["data_root"]).resolve() != Path(args.data_root).resolve():
            raise ValueError("Resume data-root differs from the original dataset. Keep the dataset path fixed.")
        for key in (
            "batch_size",
            "accum_steps",
            "max_epochs",
            "base_lr",
            "weight_decay",
            "label_order",
            "img_size",
            "seed",
            "amp",
            "num_workers",
            "deterministic",
            "clip_grad",
            "ce_weight",
            "dice_weight",
            "sampler",
            "rare_boost",
            "foreground_boost",
        ):
            if checkpoint["args"][key] != vars(args)[key]:
                raise ValueError(
                    f"Resume changes {key}: {checkpoint['args'][key]} -> {vars(args)[key]}. Use the original training settings."
                )
        _validate_resume_loss_config(checkpoint["args"], args, checkpoint.get("loss_protocol_version", 1))
        if not _configs_equal(checkpoint["model_config"], model_config) or checkpoint["split_hashes"] != split_hashes:
            raise ValueError("Resume architecture or train/val/test manifests differ from checkpoint.")
        if checkpoint["total_updates"] != total_updates:
            raise ValueError("Resume data length changes the learning-rate schedule.")
        if checkpoint.get("data_provenance", {}) != data_provenance:
            raise ValueError("Dataset preprocessing metadata changed since the checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, global_step = checkpoint["epoch"] + 1, checkpoint["global_step"]
        best_score, best_epoch, bad_epochs = (
            checkpoint["best_score"],
            checkpoint["best_epoch"],
            checkpoint["bad_epochs"],
        )
        if args.patience == 0:
            bad_epochs = 0
        stored_inclusive = checkpoint.get("best_inclusive_score")
        if stored_inclusive is not None:
            if not math.isfinite(stored_inclusive) or not 0 <= stored_inclusive <= 1:
                raise ValueError("Invalid saved best inclusive score")
            best_inclusive_score = stored_inclusive
        else:
            # Backward compatibility: old checkpoints did not persist this monitor.
            inclusive_path = layout.checkpoints / "best_inclusive.pth"
            if inclusive_path.is_file():
                inclusive_checkpoint = load_checkpoint(inclusive_path)
                if (inclusive_checkpoint.get("benchmark_data") != benchmark_data or
                        inclusive_checkpoint.get("benchmark_protocol") != benchmark_protocol):
                    raise ValueError("Best inclusive checkpoint belongs to a different benchmark")
                recovered = inclusive_validation_score(inclusive_checkpoint.get("validation_volume_metrics", {}))
                if recovered is not None:
                    best_inclusive_score = recovered
                del inclusive_checkpoint
        patience_score = checkpoint["patience_score"]
        generator.set_state(checkpoint["loader_rng"])
        restore_rng(checkpoint["rng"])
        logger.info("Resumed complete state at epoch %d, optimizer update %d", start_epoch + 1, global_step)
        del checkpoint
        del resume_checkpoint

    config_record = dict(
        loss_protocol_version=LOSS_PROTOCOL_VERSION,
        benchmark_protocol=benchmark_protocol,
        benchmark_data=benchmark_data,
        args=vars(args),
        model_config=model_config,
        class_names=CLASS_NAMES,
        data_provenance=data_provenance,
        split_hashes=split_hashes,
        parameters=sum(p.numel() for p in model.parameters()),
        torch_version=str(torch.__version__),
        numpy_version=np.__version__,
        python=sys.version,
        package_versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "scipy", "h5py", "nibabel", "ml-collections")
        },
        device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        amp_dtype=str(amp_dtype),
        total_optimizer_updates=total_updates,
    )
    write_json(layout.logs / "config.json", config_record)
    logger.info(
        "Device %s | AMP %s | parameters %s | train/val slices %d/%d | effective batch <= %d",
        device,
        amp_dtype,
        f"{config_record['parameters']:,}",
        len(datasets[0]),
        len(datasets[1]),
        args.batch_size * args.accum_steps,
    )
    logger.info(
        "Selection: validation patient-mean scar/exclusive-edema Dice (avg_pathology_dice); early stopping %s",
        f"patience={args.patience}" if args.patience else "disabled (fixed ablation budget)",
    )
    writer = step_writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(
                str(layout.tensorboard / "epochs"),
                purge_step=start_epoch + 1 if args.resume else None,
            )
            step_writer = SummaryWriter(
                str(layout.tensorboard / "updates"),
                purge_step=global_step + 1 if args.resume else None,
            )
        except ImportError:
            logger.warning("TensorBoard unavailable; CSV, JSONL and text logging remain enabled.")

    stop_epoch = min(args.max_epochs, start_epoch + args.epochs_per_run) if args.epochs_per_run else args.max_epochs
    try:
        if args.patience and bad_epochs >= args.patience:
            logger.info("Checkpoint already satisfied early stopping; no additional updates.")
            return {"best_score": best_score, "best_epoch": best_epoch, "global_step": global_step}
        for epoch in range(start_epoch, stop_epoch):
            if hasattr(criterion, "set_epoch"):
                criterion.set_epoch(epoch + 1)
            epoch_started = time.perf_counter()
            lr_start = optimizer.param_groups[0]["lr"]
            train_metrics, global_step = run_epoch(
                model,
                trainloader,
                criterion,
                device,
                amp_dtype,
                optimizer,
                scaler,
                scheduler,
                args.accum_steps,
                args.clip_grad,
                args.log_every,
                logger,
                step_writer,
                global_step,
            )
            val_metrics, _ = run_epoch(
                model,
                valloader,
                criterion,
                device,
                amp_dtype,
                log_every=args.log_every,
                logger=logger,
            )
            volume_started = time.perf_counter()
            volume_metrics = validate_volumes(model, validation_volumes, args.img_size,
                                               args.batch_size, device, amp_dtype, dataset_id=dataset_id)
            volume_seconds = time.perf_counter() - volume_started
            score = volume_metrics["avg_pathology_dice"]
            if score is None:
                raise ValueError("Pathology selection requires defined validation Dice for both scar and edema")
            if not math.isfinite(score):
                raise ValueError("Validation foreground Dice undefined: check labels and held-out patients.")
            meaningful = score > patience_score + args.min_delta
            improved = score > best_score
            bad_epochs = 0 if meaningful else bad_epochs + 1
            if meaningful:
                patience_score = score
            if improved:
                best_score, best_epoch = score, epoch
            score_inclusive = inclusive_validation_score(volume_metrics)
            inclusive_improved = score_inclusive is not None and score_inclusive > best_inclusive_score
            if inclusive_improved:
                best_inclusive_score = score_inclusive
            early_stop = bool(args.patience and bad_epochs >= args.patience)
            record = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "lr_start": lr_start,
                "lr_next": optimizer.param_groups[0]["lr"],
                "best_val_dice": best_score,
                "best_epoch": best_epoch + 1,
                "bad_epochs": bad_epochs,
                "early_stop": early_stop,
                "epoch_compute_seconds": time.perf_counter() - epoch_started,
                "val/volume_seconds": volume_seconds,
                "val/avg_pathology_inclusive_dice": score_inclusive,
            }
            record.update({f"train/{key}": value for key, value in train_metrics.items()})
            record.update({f"val/{key}": value for key, value in val_metrics.items()})
            record["val/avg_pathology_dice"] = score
            for region in ("scar", "edema"):
                for metric in ("precision", "recall"):
                    record[f"val/patient_{metric}/{region}"] = volume_metrics[region][f"mean_{metric}"]
                    for status in ("defined", "undefined"):
                        record[f"val/{metric}_{status}_cases/{region}"] = volume_metrics[region][f"{metric}_{status}_cases"]
                record[f"val/patient_dice/{region}"] = volume_metrics[region]["mean_dice"]
                record[f"val/defined_cases/{region}"] = volume_metrics[region]["dice_defined_cases"]
                record[f"val/undefined_cases/{region}"] = volume_metrics[region]["dice_undefined_cases"]
            payload = dict(
                loss_protocol_version=LOSS_PROTOCOL_VERSION,
                benchmark_protocol=benchmark_protocol,
                benchmark_data=benchmark_data,
                validation_volume_metrics=volume_metrics,
                format_version=1,
                class_names=CLASS_NAMES,
                model_config=model_config,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                epoch=epoch,
                global_step=global_step,
                total_updates=total_updates,
                best_score=best_score,
                best_epoch=best_epoch,
                best_inclusive_score=best_inclusive_score if math.isfinite(best_inclusive_score) else None,
                bad_epochs=bad_epochs,
                patience_score=patience_score,
                args=vars(args),
                data_provenance=data_provenance,
                split_hashes=split_hashes,
                rng=rng_state(),
                loader_rng=generator.get_state(),
            )
            checkpoint_started = time.perf_counter()
            if improved:
                atomic_checkpoint(layout.checkpoints / "best.pth", payload)
                logger.info(
                    "New best checkpoint: epoch %d, validation Dice %.5f -> %s",
                    epoch + 1,
                    score,
                    layout.checkpoints / "best.pth",
                )
            if inclusive_improved:
                atomic_checkpoint(layout.checkpoints / "best_inclusive.pth", payload)
                logger.info(
                    "New best inclusive checkpoint: epoch %d, (Scar+EdemaInc)/2 %.5f -> %s",
                    epoch + 1,
                    score_inclusive,
                    layout.checkpoints / "best_inclusive.pth",
                )
            # Publish the recovery point after best, so a crash cannot advance
            # the saved best_score while leaving best.pth missing or stale.
            atomic_checkpoint(layout.checkpoints / "last.pth", payload)
            if args.save_every and (epoch + 1) % args.save_every == 0:
                atomic_checkpoint(layout.checkpoints / f"epoch_{epoch + 1:04d}.pth", payload)
            record["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
            record["epoch_seconds"] = time.perf_counter() - epoch_started
            with (layout.logs / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
            csv_path = layout.logs / "metrics.csv"
            append_metrics_csv(csv_path, json_safe(record))
            if writer:
                for key, value in record.items():
                    if isinstance(value, (float, int)) and math.isfinite(value):
                        writer.add_scalar(key, value, epoch + 1)
                writer.flush()
            logger.info(
                "Epoch %d/%d | train %.5f val %.5f | val Dice %.5f IoU %.5f | lr %.3g | %.1fs | peak GPU %.0f MiB",
                epoch + 1,
                args.max_epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                score,
                val_metrics["mean_iou"],
                lr_start,
                record["epoch_seconds"],
                train_metrics["gpu_peak_allocated_mb"],
            )
            write_json(layout.logs / "summary.json", record)
            logger.info("Validation scar P/R %.4f/%.4f | edema P/R %.4f/%.4f (pixel-pooled)",
                        val_metrics["precision/scar"], val_metrics["recall/scar"],
                        val_metrics["precision/edema"], val_metrics["recall/edema"])
            if early_stop:
                logger.info("Early stopping after %d validation epochs without sufficient improvement.", bad_epochs)
                break
    except BaseException:
        logger.exception("Training interrupted/failed; last.pth contains the last completed epoch if available.")
        raise
    finally:
        if writer:
            writer.close()
        if step_writer:
            step_writer.close()
    return {"best_score": best_score, "best_epoch": best_epoch, "global_step": global_step}
