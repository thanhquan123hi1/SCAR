"""Evaluate a saved prompt-free checkpoint on held-out 3D cases."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

from ml_collections import ConfigDict
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset.data_contract import CLASS_NAMES, patient_id, read_split_names, lock_benchmark_data, resolve_label_order
from training.dataset.myops_dataset import MyopsDataset
from training.predict import predict_volume
from training.metrics.surface_distance import protocol_for_dataset, dataset_rows, summarize_dataset_rows
from training.models import model_from_config
from training.run_layout import RunLayout
from training.evaluation_logging import evaluation_logger
from training.trainer.trainer import (
    json_safe,
    load_checkpoint,
    resolve_amp,
    resolve_device,
    write_json,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--list-dir", default=None, help="defaults to saved run splits")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["test_vol", "val_vol"], default="test_vol")
    parser.add_argument("--batch-size", type=int, default=2, help="inference slices per batch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto")
    parser.add_argument(
        "--label-order",
        choices=["legacy", "canonical"],
        default=None,
        help="default: checkpoint data convention",
    )
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    from training.dataset.benchmark_profiles import DATASET_IDS
    parser.add_argument("--dataset", choices=DATASET_IDS, default=None,
                        help="Optional identity check; defaults to checkpoint dataset")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.cpu_threads < 1:
        raise ValueError("batch-size and cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    checkpoint = load_checkpoint(args.checkpoint)
    dataset_id = checkpoint["args"].get("dataset_id", "myops380")
    if args.dataset is not None and args.dataset != dataset_id:
        raise ValueError("Evaluation dataset differs from the checkpoint benchmark")
    benchmark_protocol = protocol_for_dataset(dataset_id)
    distance_unit = benchmark_protocol["distance_unit"]
    if checkpoint.get("benchmark_protocol") != benchmark_protocol or "benchmark_data" not in checkpoint:
        raise ValueError("Checkpoint dataset/metric protocol is incompatible with this benchmark")
    label_order = args.label_order or checkpoint["args"]["label_order"]
    if resolve_label_order(label_order) != checkpoint["benchmark_data"]["source_label_order"]:
        raise ValueError("Evaluation label order differs from the locked training convention")
    device = resolve_device(args.device)
    amp_dtype = resolve_amp(args.amp, device)

    config = ConfigDict(checkpoint["model_config"])
    model = model_from_config(config, img_size=checkpoint["args"]["img_size"], num_classes=4)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    checkpoint_path = Path(args.checkpoint).resolve()
    layout = RunLayout.from_checkpoint(checkpoint_path)
    saved_splits = layout.logs / "splits"
    list_dir = Path(args.list_dir) if args.list_dir else saved_splits
    output = Path(args.output_dir) if args.output_dir else layout.evaluation(args.split)
    roots = [str(Path(args.data_root) / modality / f"{args.split}_h5") for modality in ("bSSFP", "LGE", "T2w")]

    for split_name, expected in checkpoint["split_hashes"].items():
        manifest = saved_splits / f"{split_name}.txt"
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Saved {split_name} manifest was modified after training.")

    training_ids = {patient_id(name) for name in read_split_names(saved_splits, "train")}
    validation_ids = {patient_id(name) for name in read_split_names(saved_splits, "val")}
    evaluation_ids = {patient_id(name) for name in read_split_names(list_dir, args.split)}
    forbidden_ids = training_ids | validation_ids if args.split == "test_vol" else training_ids
    overlap = forbidden_ids & evaluation_ids
    if overlap:
        raise ValueError(
            f"Evaluation split overlaps patients used for training/model selection: {sorted(overlap)[:10]}"
        )

    saved_ids = set(read_split_names(saved_splits, args.split))
    if set(read_split_names(list_dir, args.split)) != saved_ids:
        raise ValueError("Evaluation patients must exactly match the saved split; subsets are not a locked benchmark")
    data_lock = lock_benchmark_data(args.data_root, saved_splits, label_order, dataset_id=dataset_id)
    if data_lock != checkpoint["benchmark_data"]:
        raise ValueError("Cache bytes or label convention changed since training")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation directory is not empty; choose a new --output-dir")

    dataset = MyopsDataset(
        *roots,
        str(list_dir),
        args.split,
        label_order=label_order,
    )
    output.mkdir(parents=True, exist_ok=True)
    with evaluation_logger(output / "test.log") as logger:
        logger.info("Checkpoint %s | split %s | device %s | AMP %s | output %s",
                    checkpoint_path, args.split, device, amp_dtype, output)
        logger.info("Inference: one forward pass per batch, argmax logits, no mask filtering")
        rows, total_seconds, total_slices = [], 0.0, 0
        for sample in dataset:
            case = sample["case_name"]
            if dataset_id != "myops380" and not sample["has_geometry"]:
                raise ValueError(f"{case}: MyoPS++ evaluation requires mm geometry of the evaluation grid")
            images = [sample[k] for k in ("image", "image1", "image2")]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction = predict_volume(
                model,
                images,
                checkpoint["args"]["img_size"],
                args.batch_size,
                device,
                amp_dtype,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            total_seconds += elapsed
            total_slices += prediction.shape[2]
            target = np.asarray(sample["label"])
            case_rows = dataset_rows(prediction, target, case, dataset_id=dataset_id, spacing=sample.get("spacing"))
            for row in case_rows:
                row["inference_seconds"] = elapsed
            rows.extend(case_rows)
            if args.save_predictions:
                np.savez_compressed(
                    output / f"{case}_pred.npz",
                    prediction=prediction,
                    class_names=np.asarray(CLASS_NAMES),
                    spacing=np.asarray(sample["spacing"]),
                    affine=np.asarray(sample["affine"]),
                    spacing_unit=sample["spacing_unit"],
                    metric_distance_unit=distance_unit,
                )
                if sample.get("has_affine", False):
                    import nibabel as nib

                    nifti = nib.Nifti1Image(prediction, np.asarray(sample["affine"]))
                    if sample.get("spacing_unit") == "mm":
                        nifti.header.set_xyzt_units("mm")
                    nib.save(nifti, output / f"{case}_pred.nii.gz")
            logger.info("%s: %d slices, %.3fs, HD95 unit=%s",
                        case, prediction.shape[2], elapsed, distance_unit)
            for row in case_rows:
                logger.info("%s | %s | Dice=%s IoU=%s Precision=%s Recall=%s HD95=%s ASD=%s",
                            case, row["region"], row["dice"], row["iou"], row["precision"],
                            row["recall"], row[f"hd95_{distance_unit}"], row[f"asd_{distance_unit}"])
        if not rows:
            raise ValueError("No evaluation cases.")
        with (output / "per_case.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = summarize_dataset_rows(rows, dataset_id=dataset_id)
        summary.update(
            checkpoint=str(checkpoint_path),
            checkpoint_epoch=checkpoint["epoch"] + 1,
            ablation=config.ablation,
            architecture=config.get("architecture", "cmspa_net"),
            split=args.split,
            case_count=len(dataset),
            inference_seconds=total_seconds,
            inference_slices_per_second=total_slices / total_seconds,
            timing_note="Includes transfer and resizing, excludes disk I/O/metrics; first case includes warmup.",
            benchmark_protocol=benchmark_protocol,
            dataset_id=dataset_id,
            inference_protocol={
                "id": "single_pass_argmax_v1",
                "forward_passes_per_batch": 1,
                "spatial_aggregation": "none",
                "mask_processing": "none",
                "image_resize": "bilinear_align_corners_false",
                "logit_resize": "bilinear_align_corners_false",
                "prediction_rule": "argmax_logits",
            },
            benchmark_data=data_lock,
            split_hashes=checkpoint["split_hashes"],
            hd95_note=("All distances are voxel distances (unit grid), never mm. Primary means exclude undefined surfaces; inspect counts. Official reproduction is separately named and retains the upstream empty-mask behavior."
                       if distance_unit == "voxel" else
                       ("Distances use preprocessed oracle ROI grid spacing in mm; GT is evaluated on that ROI, without native restoration. "
                        if benchmark_protocol.get("evaluation_grid") == "preprocessed_roi" else
                        "Distances use native orthogonal-grid spacing in mm. ") +
                       "Means exclude undefined surfaces; inspect defined/undefined counts. No author-specific empty-mask reproduction metric."),
            device=str(device),
            amp_dtype=str(amp_dtype),
        )
        write_json(output / "metrics.json", summary)
        logger.info("Summary:\n%s", json.dumps(json_safe(summary), indent=2))
        return summary


if __name__ == "__main__":
    main()
