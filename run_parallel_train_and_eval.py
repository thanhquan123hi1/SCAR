#!/usr/bin/env python3
"""Orchestrator for parallel training and testing of M2-Max-V5 and M2-Max-V6.

1. Trains M2-Max-V5 (300 epochs from ViT pretrained) and M2-Max-V6 (60 epochs
   fine-tuned from best M2-Max-V4) concurrently on CUDA.
2. Monitors both training runs with live epoch/metric logging.
3. Automatically evaluates each model on the 76-patient test benchmark
   (test_vol) across best.pth, best_inclusive.pth, and last.pth as each finishes.
4. Generates a comprehensive final benchmark comparison report.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PROJECT_ROOT = Path("/content/SCAR").resolve()
DATA_ROOT = PROJECT_ROOT / "MyoPS380_dataset" / "Processed_data"
LIST_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
V5_DIR = RUNS_DIR / "m2_max_v5"
V6_DIR = RUNS_DIR / "m2_max_v6"

V5_PRETRAINED = PROJECT_ROOT / "model" / "vit_checkpoint" / "imagenet21k" / "R50-ViT-B_16.npz"
V6_INIT_WEIGHTS = RUNS_DIR / "m2_max_v4" / "checkpoints" / "best.pth"

LOG_DIR = PROJECT_ROOT / "outputs" / "parallel_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
V5_STDOUT_LOG = LOG_DIR / "train_v5_stdout.log"
V6_STDOUT_LOG = LOG_DIR / "train_v6_stdout.log"

# Target benchmark criteria from official paper / competition
TARGETS = {
    "scar_dice": (0.7364, "Dice > 73.64%"),
    "scar_hd95": (3.66, "HD95 < 3.66 vox"),
    "edema_inc_dice": (0.7544, "Dice > 75.44%"),
    "edema_inc_hd95": (3.89, "HD95 < 3.89 vox"),
    "myo_dice": (0.8680, "Dice > 86.80%"),
}


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_latest_epoch(run_dir: Path, total_epochs: int) -> tuple[int, str]:
    # Look for logs directory
    candidates = list(run_dir.glob("logs_*"))
    if not candidates:
        return 0, "Initializing..."
    
    log_file = candidates[0] / "train.log"
    if not log_file.is_file():
        return 0, "Starting..."

    try:
        lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in reversed(lines[-50:]):
            m = re.search(r"Epoch (\d+)/(\d+) \| train ([\d\.]+) val ([\d\.]+) \| val Dice ([\d\.]+) IoU ([\d\.]+)", line)
            if m:
                ep = int(m.group(1))
                t_loss = m.group(3)
                v_loss = m.group(4)
                v_dice = m.group(5)
                return ep, f"Epoch {ep}/{total_epochs} (train_loss={t_loss}, val_loss={v_loss}, val_dice={v_dice})"
    except Exception:
        pass
    return 0, "Running..."


def evaluate_checkpoint(ckpt_path: Path, eval_dir: Path) -> dict | None:
    metrics_file = eval_dir / "metrics.json"
    if metrics_file.is_file():
        try:
            return json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    cmd = [
        sys.executable,
        "test.py",
        "--checkpoint", str(ckpt_path),
        "--data-root", str(DATA_ROOT),
        "--split", "test_vol",
        "--output-dir", str(eval_dir),
        "--batch-size", "16",
        "--device", "cuda",
        "--amp", "auto",
        "--save-predictions",
    ]
    log(f"Evaluating {ckpt_path.name} -> {eval_dir.name}...")
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        log(f"Evaluation error for {ckpt_path.name}:\n{res.stderr[-500:]}")
        return None

    if metrics_file.is_file():
        try:
            return json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def run_evaluation_for_model(model_name: str, run_dir: Path) -> list[dict]:
    ckpt_dir = run_dir / "checkpoints"
    ckpts_to_evaluate = [
        ("best", ckpt_dir / "best.pth"),
        ("best_inclusive", ckpt_dir / "best_inclusive.pth"),
        ("last", ckpt_dir / "last.pth"),
    ]

    results = []
    for label, ckpt_path in ckpts_to_evaluate:
        if not ckpt_path.is_file():
            log(f"Checkpoint {ckpt_path.name} not found, skipping.")
            continue

        eval_dir = run_dir / f"eval_{label}"
        metrics = evaluate_checkpoint(ckpt_path, eval_dir)
        if metrics:
            results.append({
                "model": model_name,
                "label": label,
                "ckpt_file": ckpt_path.name,
                "epoch": metrics.get("checkpoint_epoch", "-"),
                "scar_dice": metrics.get("scar", {}).get("mean_official_dice", metrics.get("scar", {}).get("mean_dice", 0.0)) * 100,
                "scar_hd95": metrics.get("scar", {}).get("mean_official_hd95_voxel", metrics.get("scar", {}).get("mean_hd95_voxel", float("inf"))),
                "edema_inc_dice": metrics.get("edema_inclusive", {}).get("mean_official_dice", metrics.get("edema_inclusive", {}).get("mean_dice", 0.0)) * 100,
                "edema_inc_hd95": metrics.get("edema_inclusive", {}).get("mean_official_hd95_voxel", metrics.get("edema_inclusive", {}).get("mean_hd95_voxel", float("inf"))),
                "myo_dice": metrics.get("myocardial_ring", {}).get("mean_dice", 0.0) * 100,
                "edema_dice": metrics.get("edema", {}).get("mean_official_dice", metrics.get("edema", {}).get("mean_dice", 0.0)) * 100,
                "empty_count": (
                    metrics.get("edema", {}).get("status_counts", {}).get("prediction_empty", 0) +
                    metrics.get("scar", {}).get("status_counts", {}).get("prediction_empty", 0) +
                    metrics.get("edema_inclusive", {}).get("status_counts", {}).get("prediction_empty", 0)
                ),
            })
    return results


def format_results_markdown(all_results: list[dict]) -> str:
    lines = []
    lines.append("## 📊 TEST BENCHMARK RESULTS (76-Case Official Test Split)\n")
    lines.append("| Model | Checkpoint | Epoch | Scar Dice (%) | Scar HD95 (vox) | Edema-Inc Dice (%) | Edema-Inc HD95 (vox) | Myo Ring Dice (%) | Empty Masks |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in all_results:
        s_d_pass = "✅ " if r["scar_dice"] > TARGETS["scar_dice"][0] * 100 else ""
        s_h_pass = "✅ " if r["scar_hd95"] < TARGETS["scar_hd95"][0] else ""
        e_d_pass = "✅ " if r["edema_inc_dice"] > TARGETS["edema_inc_dice"][0] * 100 else ""
        e_h_pass = "✅ " if r["edema_inc_hd95"] < TARGETS["edema_inc_hd95"][0] else ""
        m_d_pass = "✅ " if r["myo_dice"] > TARGETS["myo_dice"][0] * 100 else ""
        empty_pass = "✅ " if r["empty_count"] == 0 else "⚠️ "

        lines.append(
            f"| **{r['model']}** | `{r['label']}` | {r['epoch']} | "
            f"{s_d_pass}{r['scar_dice']:.2f}% | {s_h_pass}{r['scar_hd95']:.2f} | "
            f"{e_d_pass}{r['edema_inc_dice']:.2f}% | {e_h_pass}{r['edema_inc_hd95']:.2f} | "
            f"{m_d_pass}{r['myo_dice']:.2f}% | {empty_pass}{r['empty_count']} |"
        )
    lines.append("\n*Targets: Scar Dice > 73.64%, Scar HD95 < 3.66 vox, Edema-Inc Dice > 75.44%, Edema-Inc HD95 < 3.89 vox, Myo Dice > 86.80%, Empty Masks = 0*")
    return "\n".join(lines)


def main():
    log("===================================================================")
    log("STARTING PARALLEL RETRAINING OF M2-MAX-V5 AND M2-MAX-V6")
    log("===================================================================")
    log(f"Project Root: {PROJECT_ROOT}")
    log(f"Data Root: {DATA_ROOT}")
    log(f"V5 Pretrained: {V5_PRETRAINED}")
    log(f"V6 Init Weights: {V6_INIT_WEIGHTS}")

    # Build V5 command
    v5_cmd = [
        sys.executable, "train.py",
        "--config", "training/config/models/m2_max_v5.yaml",
        "--ablation", "M2-MAX-V5",
        "--data-root", str(DATA_ROOT),
        "--list-dir", str(LIST_DIR),
        "--output-dir", str(V5_DIR),
        "--epochs", "300",
        "--batch-size", "16",
        "--lr", "0.0003",
        "--weight-decay", "0.0001",
        "--clip-grad", "1.0",
        "--ce-weight", "0.5",
        "--dice-weight", "0.5",
        "--aar-weight", "0.4",
        "--scar-weight", "0.1",
        "--dice-class-weights", "0.0", "1.0", "2.0", "2.0",
        "--ce-class-weights", "0.2", "1.0", "2.0", "2.0",
        "--sampler", "rare",
        "--rare-boost", "2.0",
        "--foreground-boost", "1.3",
        "--pretrained", str(V5_PRETRAINED),
        "--device", "cuda",
        "--amp", "auto",
        "--num-workers", "4",
        "--save-every", "0",
    ]

    # Build V6 command
    v6_cmd = [
        sys.executable, "train.py",
        "--config", "training/config/models/m2_max_v6.yaml",
        "--ablation", "M2-MAX-V6",
        "--data-root", str(DATA_ROOT),
        "--list-dir", str(LIST_DIR),
        "--output-dir", str(V6_DIR),
        "--epochs", "60",
        "--batch-size", "16",
        "--lr", "0.00015",
        "--weight-decay", "0.0001",
        "--clip-grad", "1.0",
        "--ce-weight", "0.5",
        "--dice-weight", "0.5",
        "--aar-weight", "0.4",
        "--scar-weight", "0.08",
        "--wall-weight", "0.2",
        "--dice-class-weights", "0.0", "1.0", "2.0", "2.0",
        "--ce-class-weights", "0.2", "1.0", "2.0", "2.0",
        "--sampler", "rare",
        "--rare-boost", "2.0",
        "--foreground-boost", "1.3",
        "--init-weights", str(V6_INIT_WEIGHTS),
        "--device", "cuda",
        "--amp", "auto",
        "--num-workers", "4",
        "--save-every", "0",
    ]

    log(f"Launching M2-MAX-V5 process (Stdout -> {V5_STDOUT_LOG})...")
    v5_out_f = open(V5_STDOUT_LOG, "w")
    v5_proc = subprocess.Popen(v5_cmd, cwd=str(PROJECT_ROOT), stdout=v5_out_f, stderr=subprocess.STDOUT)
    log(f"M2-MAX-V5 PID: {v5_proc.pid}")

    log(f"Launching M2-MAX-V6 process (Stdout -> {V6_STDOUT_LOG})...")
    v6_out_f = open(V6_STDOUT_LOG, "w")
    v6_proc = subprocess.Popen(v6_cmd, cwd=str(PROJECT_ROOT), stdout=v6_out_f, stderr=subprocess.STDOUT)
    log(f"M2-MAX-V6 PID: {v6_proc.pid}")

    v5_done = False
    v6_done = False
    v5_results: list[dict] = []
    v6_results: list[dict] = []

    start_time = time.time()
    last_print = 0

    try:
        while not (v5_done and v6_done):
            time.sleep(15)
            elapsed = time.time() - start_time

            # Check V5
            if not v5_done:
                poll_v5 = v5_proc.poll()
                if poll_v5 is not None:
                    v5_done = True
                    v5_out_f.close()
                    log(f"🎉 M2-MAX-V5 completed training with returncode {poll_v5}! Elapsed: {elapsed/60:.1f}m")
                    if poll_v5 == 0:
                        log("Running comprehensive evaluation on M2-MAX-V5 checkpoints...")
                        v5_results = run_evaluation_for_model("M2-MAX-V5", V5_DIR)
                    else:
                        log(f"ERROR: V5 failed! Check {V5_STDOUT_LOG}")

            # Check V6
            if not v6_done:
                poll_v6 = v6_proc.poll()
                if poll_v6 is not None:
                    v6_done = True
                    v6_out_f.close()
                    log(f"🎉 M2-MAX-V6 completed training with returncode {poll_v6}! Elapsed: {elapsed/60:.1f}m")
                    if poll_v6 == 0:
                        log("Running comprehensive evaluation on M2-MAX-V6 checkpoints...")
                        v6_results = run_evaluation_for_model("M2-MAX-V6", V6_DIR)
                    else:
                        log(f"ERROR: V6 failed! Check {V6_STDOUT_LOG}")

            # Periodic progress reporting every 30s
            if time.time() - last_print > 30 and not (v5_done and v6_done):
                last_print = time.time()
                _, v5_status = parse_latest_epoch(V5_DIR, 300)
                _, v6_status = parse_latest_epoch(V6_DIR, 60)
                v5_str = "FINISHED" if v5_done else v5_status
                v6_str = "FINISHED" if v6_done else v6_status
                log(f"[Elapsed {elapsed/60:.1f}m] V5: {v5_str} | V6: {v6_str}")

        total_elapsed = time.time() - start_time
        log(f"===================================================================")
        log(f"ALL PARALLEL TRAINING AND TESTING COMPLETED in {total_elapsed/60:.1f} minutes!")
        log(f"===================================================================")

        all_results = v5_results + v6_results
        report_md = format_results_markdown(all_results)
        print("\n" + report_md + "\n", flush=True)

        final_report_file = PROJECT_ROOT / "outputs" / "FINAL_PARALLEL_TRAIN_TEST_REPORT.md"
        final_report_file.write_text(report_md, encoding="utf-8")
        log(f"Saved final report to: {final_report_file}")

    finally:
        if not v5_proc.poll():
            v5_out_f.close()
        if not v6_proc.poll():
            v6_out_f.close()


if __name__ == "__main__":
    main()
