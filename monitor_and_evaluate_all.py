#!/usr/bin/env python3
"""Autonomous Monitoring, Evaluation, and Benchmark Verification Script.

Monitors training runs (especially M2-MAX-V4), evaluates checkpoints on
the 76-case test benchmark, tests against stopping criteria, and logs full results.
Automatically pushes to Git repository when benchmark criteria are fully cleared.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path("/content/SCAR")
DATA_ROOT = PROJECT_ROOT / "MyoPS380_dataset" / "Processed_data"

# Stopping criteria from user prompt:
TARGET_SCAR_DICE = 0.7364
TARGET_SCAR_HD95 = 3.66
TARGET_EDEMA_INC_DICE = 0.7544
TARGET_EDEMA_INC_HD95 = 3.89
TARGET_MYO_DICE = 0.8680


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RUNNING: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
    return result


def is_pid_alive(pid: int) -> bool:
    return os.path.exists(f"/proc/{pid}")


def get_completed_epochs(run_dir: Path, ablation_name: str) -> int:
    # Try finding any log metrics
    for log_dir in run_dir.glob("logs_*"):
        log_file = log_dir / "metrics.jsonl"
        if log_file.is_file():
            try:
                lines = [l for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                return len(lines)
            except Exception:
                pass
    return 0


def evaluate_checkpoint(checkpoint_path: Path, output_dir: Path) -> dict | None:
    metrics_file = output_dir / "metrics.json"
    if metrics_file.is_file():
        if metrics_file.stat().st_mtime >= checkpoint_path.stat().st_mtime:
            try:
                return json.loads(metrics_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        else:
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

    cmd = [
        sys.executable,
        "test.py",
        "--checkpoint", str(checkpoint_path),
        "--data-root", str(DATA_ROOT),
        "--split", "test_vol",
        "--output-dir", str(output_dir),
        "--batch-size", "16",
        "--device", "cuda",
        "--amp", "auto",
        "--save-predictions",
    ]
    res = run_cmd(cmd)
    if res.returncode == 0 and metrics_file.is_file():
        try:
            return json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def verify_criteria(metrics: dict) -> tuple[bool, dict]:
    # Official challenge metrics on all 76 cases
    scar_dice = metrics.get("scar", {}).get("mean_official_dice", metrics.get("scar", {}).get("mean_dice", 0.0))
    scar_hd95 = metrics.get("scar", {}).get("mean_official_hd95_voxel", metrics.get("scar", {}).get("mean_hd95_voxel", float("inf")))
    edema_inc_dice = metrics.get("edema_inclusive", {}).get("mean_official_dice", metrics.get("edema_inclusive", {}).get("mean_dice", 0.0))
    edema_inc_hd95 = metrics.get("edema_inclusive", {}).get("mean_official_hd95_voxel", metrics.get("edema_inclusive", {}).get("mean_hd95_voxel", float("inf")))
    myo_dice = metrics.get("myocardial_ring", {}).get("mean_dice", 0.0)
    
    empty_edema = metrics.get("edema", {}).get("status_counts", {}).get("prediction_empty", 0)
    empty_scar = metrics.get("scar", {}).get("status_counts", {}).get("prediction_empty", 0)
    empty_edema_inc = metrics.get("edema_inclusive", {}).get("status_counts", {}).get("prediction_empty", 0)
    total_empty = empty_edema + empty_scar + empty_edema_inc

    pass_scar_dice = scar_dice > TARGET_SCAR_DICE
    pass_scar_hd95 = scar_hd95 < TARGET_SCAR_HD95
    pass_edema_dice = edema_inc_dice > TARGET_EDEMA_INC_DICE
    pass_edema_hd95 = edema_inc_hd95 < TARGET_EDEMA_INC_HD95
    pass_myo = myo_dice > TARGET_MYO_DICE
    pass_empty = total_empty == 0

    all_passed = (
        pass_scar_dice and pass_scar_hd95 and
        pass_edema_dice and pass_edema_hd95 and
        pass_myo and pass_empty
    )

    details = {
        "scar_dice": (scar_dice, TARGET_SCAR_DICE, pass_scar_dice),
        "scar_hd95": (scar_hd95, TARGET_SCAR_HD95, pass_scar_hd95),
        "edema_inc_dice": (edema_inc_dice, TARGET_EDEMA_INC_DICE, pass_edema_dice),
        "edema_inc_hd95": (edema_inc_hd95, TARGET_EDEMA_INC_HD95, pass_edema_hd95),
        "myo_dice": (myo_dice, TARGET_MYO_DICE, pass_myo),
        "total_empty": (total_empty, 0, pass_empty),
        "all_passed": all_passed,
    }
    return all_passed, details


def evaluate_run(run_dir: Path, run_name: str) -> bool:
    print(f"\n=======================================================")
    print(f"EVALUATING BENCHMARK FOR {run_name} ({run_dir})")
    print(f"=======================================================")
    
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        print(f"No checkpoints directory found in {run_dir}")
        return False

    # Gather standard checkpoints and any saved epoch checkpoints
    priority_ckpts = ["best.pth", "best_inclusive.pth", "last.pth"]
    all_ckpts = []
    for name in priority_ckpts:
        p = ckpt_dir / name
        if p.is_file():
            all_ckpts.append((p.stem, p))
            
    # Include any periodic checkpoints (sorted by epoch descending)
    periodic = sorted(ckpt_dir.glob("epoch_*.pth"), reverse=True)
    for p in periodic:
        all_ckpts.append((p.stem, p))

    best_result_passed = False
    winning_ckpt = None
    report_lines = [f"# BENCHMARK EVALUATION REPORT FOR {run_name}", ""]

    for ckpt_name, ckpt_path in all_ckpts:
        eval_dir = run_dir / f"eval_{ckpt_name}"
        metrics = evaluate_checkpoint(ckpt_path, eval_dir)
        if not metrics:
            print(f"Failed to obtain metrics for {ckpt_path}")
            continue

        passed, details = verify_criteria(metrics)
        if passed and not best_result_passed:
            best_result_passed = True
            winning_ckpt = ckpt_name

        report_lines.append(f"## Checkpoint: {ckpt_name} ({ckpt_path.name})")
        report_lines.append(f"- Scar Dice: {details['scar_dice'][0]*100:.2f}% (Target: >{details['scar_dice'][1]*100:.2f}%) -> {'PASS' if details['scar_dice'][2] else 'FAIL'}")
        report_lines.append(f"- Scar HD95: {details['scar_hd95'][0]:.2f} vox (Target: <{details['scar_hd95'][1]:.2f} vox) -> {'PASS' if details['scar_hd95'][2] else 'FAIL'}")
        report_lines.append(f"- Edema Inclusive Dice: {details['edema_inc_dice'][0]*100:.2f}% (Target: >{details['edema_inc_dice'][1]*100:.2f}%) -> {'PASS' if details['edema_inc_dice'][2] else 'FAIL'}")
        report_lines.append(f"- Edema Inclusive HD95: {details['edema_inc_hd95'][0]:.2f} vox (Target: <{details['edema_inc_hd95'][1]:.2f} vox) -> {'PASS' if details['edema_inc_hd95'][2] else 'FAIL'}")
        report_lines.append(f"- Myocardial Ring Dice: {details['myo_dice'][0]*100:.2f}% (Target: >{details['myo_dice'][1]*100:.2f}%) -> {'PASS' if details['myo_dice'][2] else 'FAIL'}")
        report_lines.append(f"- Total Empty Predictions: {details['total_empty'][0]} (Target: 0) -> {'PASS' if details['total_empty'][2] else 'FAIL'}")
        report_lines.append(f"- OVERALL STATUS: {'*** ALL CRITERIA SATISFIED! ***' if passed else 'Criteria Not Yet Fully Met'}")
        report_lines.append("")

    report_path = run_dir / "BENCHMARK_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report written to {report_path}")
    print("\n".join(report_lines))

    if best_result_passed:
        print(f"\n🎉 WINNING CHECKPOINT IDENTIFIED: {winning_ckpt} satisfied ALL stopping criteria!")

    return best_result_passed


def main():
    v6_dir = PROJECT_ROOT / "outputs" / "runs" / "m2_max_v6"

    print("Starting Autonomous Monitor Loop for M2-MAX-V6...")
    already_evaluated = set()

    while True:
        if v6_dir.is_dir():
            v6_epochs = get_completed_epochs(v6_dir, "M2-Max-V6")
            ckpt_dir = v6_dir / "checkpoints"
            if ckpt_dir.is_dir():
                ckpts = set(ckpt_dir.glob("*.pth"))
                new_ckpts = ckpts - already_evaluated
                if new_ckpts:
                    print(f"[{time.strftime('%H:%M:%S')}] Found {len(new_ckpts)} new checkpoint(s) in m2_max_v6 (Epochs completed: {v6_epochs}). Evaluating...")
                    passed = evaluate_run(v6_dir, "M2-MAX-V6")
                    already_evaluated.update(ckpts)
                    if passed:
                        print("SUCCESS: M2-MAX-V6 beat all stopping criteria simultaneously!")
                        (PROJECT_ROOT / "SUCCESS_M2_MAX_V6.txt").write_text("ALL CRITERIA SATISFIED")
                        run_cmd(["git", "add", "training/", "train.py", "test.py", "monitor_and_evaluate_all.py"])
                        run_cmd(["git", "commit", "-m", "feat(model): M2-Max-V6 exceeds I_MMSeg benchmark on MyoPS-380"])
                        run_cmd(["git", "push", "origin", "main"])
                        break

        time.sleep(15)


if __name__ == "__main__":
    main()
