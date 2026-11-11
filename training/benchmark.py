"""Synthetic SSPANet/M2-Plus benchmarks; no dataset or long training required.

Run with ``python -m training.benchmark --help``. CUDA events measure GPU latency;
CPU mode is for software checks. Training measurements include forward, loss,
backward and AdamW, but exclude loading, augmentation, metrics and scheduling.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
from pathlib import Path
import platform
import statistics
import time

from ml_collections import ConfigDict
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

from training.loss.losses import SegmentationLoss
from training.models import model_from_config
from training.models.modules.sspanet import LEGACY_SSPA_GATE, SSPA_GATE_MODES
from training.optimization import build_optimizer
from training.provenance import code_provenance
from training.train import model_config_from_args, parse_args as training_args
from training.trainer.trainer import load_checkpoint, resolve_device, seed_everything, write_json


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", default=None, help="model YAML; defaults to m2_plus_paper_formula")
    source.add_argument("--checkpoint", help="benchmark saved architecture and weights exactly")
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--dataset", default=None, help="recipe profile only; no images are loaded")
    parser.add_argument("--variants", nargs="+", choices=SSPA_GATE_MODES, default=None)
    parser.add_argument("--scope", choices=("module", "model", "both"), default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-a100", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true", help="default: strict FP32, TF32 disabled")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="inference batch")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--output", default="outputs/benchmarks/sspanet/efficiency_metrics.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def counted_flops(module, inputs):
    """Count conv/matrix FLOPs, forcing unfused attention only for this pass.

    Pooling, normalization, statistics, softmax and other elementwise operations
    are excluded. This is a consistent arithmetic count, not a latency proxy.
    Profiling uses CPU operator events even when tensors are on CUDA.
    """
    module.eval()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
            output = module(*inputs)
    if not torch.isfinite(output).all():
        raise FloatingPointError("Non-finite benchmark forward output")
    supported = {"aten::conv2d", "aten::mm", "aten::addmm", "aten::bmm"}
    by_op = {event.key: int(event.flops) for event in prof.key_averages()
             if event.key in supported and event.flops}
    flops = sum(by_op.values())
    return {"flops": flops, "macs": flops // 2, "flops_by_op": by_op,
            "output_shape": list(output.shape),
            "definition": "Conv2d and matrix multiplications, including attention QK/AV; "
                          "one MAC = two FLOPs; excludes pooling/norm/statistics/softmax/elementwise; "
                          "SDPA math backend forced only for counting"}


def measure(operation, device, warmup, iterations, repeats, batch_size):
    for _ in range(warmup):
        operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    durations, repeat_means = [], []
    for _ in range(repeats):
        samples = []
        for _ in range(iterations):
            if device.type == "cuda":
                start.record()
                operation()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            else:
                started = time.perf_counter()
                operation()
                samples.append((time.perf_counter() - started) * 1000)
        durations.extend(samples)
        repeat_means.append(statistics.mean(samples))
    mean = statistics.mean(durations)
    return {"mean_ms": mean, "median_ms": statistics.median(durations),
            "p95_ms": float(np.percentile(durations, 95)),
            "std_ms": statistics.pstdev(durations), "repeat_mean_ms": repeat_means,
            "samples_per_second": batch_size * 1000 / mean if mean > 0 else None,
            "peak_allocated_mib": (torch.cuda.max_memory_allocated(device) / 2**20
                                   if device.type == "cuda" else None),
            "peak_reserved_mib": (torch.cuda.max_memory_reserved(device) / 2**20
                                  if device.type == "cuda" else None)}


def make_criterion(args):
    keys = ("ce_weight", "dice_weight", "aar_weight", "scar_weight", "dice_class_weights",
            "ce_class_weights", "pathology_reduction", "negative_slice_handling",
            "masked_negative_slices", "batch_dice", "include_background", "dice_normalization")
    return SegmentationLoss(**{key: getattr(args, key) for key in keys})


def benchmark_subject(module, scope, channels, img_size, args, recipe, device):
    module = module.to(device=device, dtype=torch.float32)
    count = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

    def inputs_for(batch):
        if scope == "module":
            return (torch.rand(batch, channels, img_size // 16, img_size // 16, device=device),)
        return tuple(torch.rand(batch, 1, img_size, img_size, device=device) for _ in range(3))

    inputs = inputs_for(args.batch_size)
    result = {"params_total": count, "params_trainable": trainable,
              "input_shapes": [list(x.shape) for x in inputs],
              "arithmetic": counted_flops(module, inputs)}
    module.eval()
    with torch.inference_mode():
        result["inference"] = measure(lambda: module(*inputs), device, args.warmup,
                                      args.iterations, args.repeats, args.batch_size)
    del inputs
    if not args.skip_training:
        module.train()
        inputs = inputs_for(args.train_batch_size)
        # Include upstream input gradients for an isolated bottleneck, including identity.
        if scope == "module":
            inputs[0].requires_grad_(True)
        optimizer = (build_optimizer(module, recipe.base_lr, recipe.weight_decay)
                     if trainable else None)
        criterion = make_criterion(recipe) if scope == "model" else None
        target = (torch.randint(0, 4, (args.train_batch_size, img_size, img_size), device=device)
                  if scope == "model" else None)
        last_loss = None

        def step():
            nonlocal last_loss
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if scope == "module":
                inputs[0].grad = None
            output = module(*inputs)
            loss = criterion(output, target)["loss"] if criterion else output.square().mean()
            loss.backward()
            if optimizer is not None:
                optimizer.step()
            last_loss = loss.detach()

        result["training_input_shapes"] = [list(x.shape) for x in inputs]
        result["training_step"] = measure(step, device, args.warmup, args.iterations,
                                          args.repeats, args.train_batch_size)
        if not torch.isfinite(last_loss):
            raise FloatingPointError("Non-finite synthetic training loss")
        result["training_step"]["loss"] = ("configured SegmentationLoss" if criterion
                                           else "mean(output**2), including input gradient")
        result["training_step"]["optimizer"] = "AdamW" if optimizer else None
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("batch_size", "train_batch_size", "warmup", "iterations", "repeats", "cpu_threads"):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Benchmark output already exists: {output}")
    cli = ["--config", args.config or "m2_plus_paper_formula"]
    if args.base_config:
        cli += ["--base-config", args.base_config]
    if args.dataset:
        cli += ["--dataset", args.dataset]
    recipe, merged = training_args(cli)
    config = model_config_from_args(recipe, merged)
    checkpoint = None
    source = {"kind": "random_initialization", "config": args.config or "m2_plus_paper_formula"}
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint)
        config = ConfigDict(checkpoint["model_config"])
        vars(recipe).update(checkpoint.get("args", {}))
        with Path(args.checkpoint).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        source = {"kind": "checkpoint", "path": str(Path(args.checkpoint).resolve()), "sha256": digest}
    img_size = args.img_size if args.img_size is not None else recipe.img_size
    if img_size < 32 or img_size % 16:
        parser.error("img-size must be >=32 and divisible by 16")
    variants = args.variants or [config.get("sspanet_gate", LEGACY_SSPA_GATE)]
    if len(variants) != len(set(variants)):
        parser.error("variants must not contain duplicates")
    if checkpoint and variants != [config.get("sspanet_gate", LEGACY_SSPA_GATE)]:
        parser.error("A checkpoint must be benchmarked with its saved SSPANet variant")
    device = resolve_device(args.device)
    if device.type == "cuda":
        # torch.cuda.set_device rejects an index-less torch.device('cuda').
        device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    if args.require_a100 and (device.type != "cuda" or "A100" not in device_name.upper()):
        raise RuntimeError("--require-a100 requires an NVIDIA A100 device")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    gpu = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    report = {"protocol": "scar_sspanet_synthetic_fp32_v1", "status": "running", "source": source,
              "device": str(device), "device_name": device_name, "python": platform.python_version(),
              "gpu_properties": ({"total_memory_bytes": gpu.total_memory,
                                  "compute_capability": [gpu.major, gpu.minor],
                                  "multiprocessor_count": gpu.multi_processor_count} if gpu else None),
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "precision": "float32", "amp": False,
              "tf32": args.allow_tf32, "seed": args.seed, "warmup": args.warmup,
              "inference_batch_size": args.batch_size, "train_batch_size": args.train_batch_size,
              "cpu_threads": args.cpu_threads,
              "iterations": args.iterations, "repeats": args.repeats,
              "timing_method": "cuda_events" if device.type == "cuda" else "cpu_perf_counter",
              "data": "synthetic uniform images/features and uniform random four-class targets",
              "training_scope": "forward + loss + backward + AdamW at constant LR; "
                                "no data loading/augmentation/metrics/scheduler; not epoch time or Dice",
              "training_recipe": {key: getattr(recipe, key) for key in (
                  "base_lr", "weight_decay", "ce_weight", "dice_weight", "dice_class_weights",
                  "ce_class_weights", "negative_slice_handling", "include_background",
                  "dice_normalization", "aar_weight", "scar_weight", "pathology_reduction")},
              "code_provenance": code_provenance(), "results": []}
    scopes = ("module", "model") if args.scope == "both" else (args.scope,)
    output.parent.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        selected = deepcopy(config)
        selected.sspanet_gate = variant
        for scope in scopes:
            seed_everything(args.seed, deterministic=True)
            module = model_from_config(selected, img_size=img_size)
            if checkpoint:
                module.load_state_dict(checkpoint["model"], strict=True)
            resolved = module.config.to_dict()
            channels = 16 * int(64 * resolved["resnet"]["width_factor"])
            if scope == "module":
                module = module.sspanet_cine
            try:
                result = benchmark_subject(module, scope, channels, img_size, args, recipe, device)
                report["results"].append({"variant": variant, "scope": scope,
                                          "model_config": resolved, **result})
                write_json(output, report)
                print(f"{variant} / {scope}: {result['params_total']:,} params; "
                      f"inference {result['inference']['mean_ms']:.3f} ms", flush=True)
            except Exception as exc:
                report["status"] = "failed"
                report["error"] = {"variant": variant, "scope": scope,
                                   "type": type(exc).__name__, "message": str(exc)}
                write_json(output, report)
                raise
            finally:
                del module
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    report["status"] = "complete"
    write_json(output, report)
    print(f"Saved {output.resolve()}")
    return report


if __name__ == "__main__":
    main()
