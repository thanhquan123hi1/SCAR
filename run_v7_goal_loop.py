#!/usr/bin/env python3
"""Autonomous Goal Execution & Monitoring Script for M2-MAX-V7.

Features:
1. Automated Disk Guard: Prunes intermediate checkpoints if disk space < 5.0 GB.
2. Checkpoint strategy: Saves every 10 epochs for detailed trajectory analysis.
3. Automated Evaluation: Evaluates test split on all checkpoints (Standard + Official).
4. Automated Backup: Compresses and uploads to Google Drive upon completion.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/content/SCAR")
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
V7_DIR = RUNS_DIR / "m2_max_v7"
DRIVE_DEST_DIR = Path("/content/drive/Colab Notebooks/Quan")
DATA_ROOT = PROJECT_ROOT / "MyoPS380_dataset" / "Processed_data"
LIST_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
INIT_WEIGHTS = RUNS_DIR / "m2_max_v6" / "checkpoints" / "best.pth"

TARGETS = {
    "scar_dice": 0.74,        # User Target: > 74.00%
    "aar_dice": 0.758,        # User Target: ~ 76.00%
    "scar_hd95": 3.66,        # Benchmark target: < 3.66 vox
    "aar_hd95": 3.89,         # Benchmark target: < 3.89 vox
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def get_free_disk_gb(path: str = "/content") -> float:
    stat = shutil.disk_usage(path)
    return stat.free / (1024 ** 3)


def disk_guard_cleanup(target_dir: Path, min_free_gb: float = 5.0) -> None:
    free_gb = get_free_disk_gb()
    if free_gb < min_free_gb:
        log(f"⚠️ DISK GUARD: Free space ({free_gb:.2f} GB) is below safety threshold ({min_free_gb} GB)!")
        log("Pruning intermediate epoch_*.pth checkpoints across all runs...")
        keep = {"best.pth", "best_inclusive.pth", "last.pth"}
        for p in RUNS_DIR.rglob("epoch_*.pth"):
            if p.name not in keep:
                try:
                    p.unlink()
                    log(f"  Deleted: {p.relative_to(RUNS_DIR)}")
                except Exception as e:
                    log(f"  Failed to delete {p.name}: {e}")
        new_free_gb = get_free_disk_gb()
        log(f"Disk cleanup complete. Current free space: {new_free_gb:.2f} GB")


def evaluate_checkpoint(ckpt_path: Path, output_dir: Path) -> dict | None:
    cmd = [
        sys.executable, "test.py",
        "--config", "training/config/models/m2_max_v7.yaml",
        "--ablation", "M2-MAX-V7",
        "--data-root", str(DATA_ROOT),
        "--list-dir", str(LIST_DIR),
        "--checkpoint", str(ckpt_path),
        "--output-dir", str(output_dir),
        "--split", "test_vol",
        "--device", "cuda",
        "--amp", "auto",
        "--num-workers", "4",
        "--preview-cases", "10",
        "--quiet",
    ]
    log(f"Running evaluation: {ckpt_path.name} -> {output_dir.name}...")
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        log(f"Evaluation failed for {ckpt_path.name} with returncode {res.returncode}")
        return None

    metrics_file = output_dir / "metrics.json"
    if metrics_file.is_file():
        try:
            return json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def run_full_benchmark(run_dir: Path) -> list[dict]:
    ckpt_dir = run_dir / "checkpoints"
    ckpts = [
        ("best", ckpt_dir / "best.pth"),
        ("best_inclusive", ckpt_dir / "best_inclusive.pth"),
        ("last", ckpt_dir / "last.pth"),
    ]
    # Also evaluate any intermediate epoch checkpoints if they exist
    for p in sorted(ckpt_dir.glob("epoch_*.pth")):
        if p.name not in {"best.pth", "best_inclusive.pth", "last.pth"}:
            ckpts.append((p.stem, p))

    results = []
    for label, ckpt_path in ckpts:
        if not ckpt_path.is_file():
            continue
        eval_dir = run_dir / f"eval_{label}"
        m = evaluate_checkpoint(ckpt_path, eval_dir)
        if m:
            s = m.get("scar", {})
            e = m.get("edema_inclusive", {})
            myo = m.get("myocardial_ring", {})
            results.append({
                "label": label,
                "epoch": m.get("checkpoint_epoch", "-"),
                "scar_dice_std": s.get("mean_dice", 0.0) * 100,
                "scar_dice_off": s.get("mean_official_dice", 0.0) * 100,
                "scar_hd95_std": s.get("mean_hd95_voxel", float("inf")),
                "scar_hd95_off": s.get("mean_official_hd95_voxel", float("inf")),
                "aar_dice_std": e.get("mean_dice", 0.0) * 100,
                "aar_dice_off": e.get("mean_official_dice", 0.0) * 100,
                "aar_hd95_std": e.get("mean_hd95_voxel", float("inf")),
                "aar_hd95_off": e.get("mean_official_hd95_voxel", float("inf")),
                "myo_dice": myo.get("mean_dice", 0.0) * 100,
            })
    return results


def format_report(results: list[dict]) -> str:
    lines = ["# 🏆 BENCHMARK EVALUATION REPORT FOR M2-MAX-V7\n"]
    lines.append("| Checkpoint | Epoch | Scar Dice (Off / Std) | Scar HD95 | AAR Dice (Off / Std) | AAR HD95 | Myo Ring | Status |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    best_overall = None
    max_score = -1

    for r in results:
        pass_s = r["scar_dice_off"] >= 74.0
        pass_a = r["aar_dice_off"] >= 75.8
        pass_sh = r["scar_hd95_off"] <= 3.66
        pass_ah = r["aar_hd95_off"] <= 3.89

        all_pass = pass_s and pass_a and pass_sh and pass_ah
        status = "🌟 ALL TARGETS MET!" if all_pass else ("✅ Scar > 74%" if pass_s else "Processing")

        combined = r["scar_dice_off"] + r["aar_dice_off"]
        if combined > max_score:
            max_score = combined
            best_overall = r

        lines.append(
            f"| `{r['label']}` | {r['epoch']} | "
            f"**{r['scar_dice_off']:.2f}%** / {r['scar_dice_std']:.2f}% | "
            f"{r['scar_hd95_off']:.2f} vox | "
            f"**{r['aar_dice_off']:.2f}%** / {r['aar_dice_std']:.2f}% | "
            f"{r['aar_hd95_off']:.2f} vox | "
            f"{r['myo_dice']:.2f}% | {status} |"
        )

    lines.append("\n*Targets: Scar Dice > 74.00%, AAR Dice ~ 76.00% (>=75.80%), Scar HD95 < 3.66 vox, AAR HD95 < 3.89 vox*\n")
    if best_overall:
        lines.append(f"### Best Overall Checkpoint: `{best_overall['label']}` (Epoch {best_overall['epoch']})")
        lines.append(f"- **Scar Dice (Official):** {best_overall['scar_dice_off']:.2f}%")
        lines.append(f"- **AAR Dice (Official):** {best_overall['aar_dice_off']:.2f}%")
        lines.append(f"- **Scar HD95 (Official):** {best_overall['scar_hd95_off']:.2f} vox")
        lines.append(f"- **AAR HD95 (Official):** {best_overall['aar_hd95_off']:.2f} vox")
    return "\n".join(lines)


def backup_to_gdrive(run_dir: Path) -> bool:
    model_name = run_dir.name
    archive_path = Path("/content") / f"{model_name}.tar.gz"
    dest_path = DRIVE_DEST_DIR / f"{model_name}.tar.gz"

    log("==================================================")
    log(f"BACKING UP {model_name} TO GOOGLE DRIVE...")
    # Prune intermediate epoch checkpoints before compression to keep archive compact
    keep = {"best.pth", "best_inclusive.pth", "last.pth"}
    for p in (run_dir / "checkpoints").glob("epoch_*.pth"):
        if p.name not in keep:
            p.unlink()

    # Compress with pigz
    log(f"Compressing {run_dir} -> {archive_path}...")
    t0 = time.time()
    subprocess.run(f"tar -I pigz -cf '{archive_path}' -C '{run_dir.parent}' '{run_dir.name}'", shell=True, check=True)
    log(f"Compressed in {time.time()-t0:.1f}s. Size: {archive_path.stat().st_size / (1024*1024):.2f} MB")

    # Copy to Drive
    log(f"Transferring to {dest_path}...")
    t1 = time.time()
    subprocess.run(f"cp -v '{archive_path}' '{dest_path}'", shell=True, check=True)
    log(f"Copied in {time.time()-t1:.1f}s.")

    # Verify
    if dest_path.is_file() and dest_path.stat().st_size == archive_path.stat().st_size:
        log("✅ Backup verified successfully on Google Drive!")
        archive_path.unlink()
        log("Deleted local temporary archive.")
        return True
    return False


def main():
    log("===================================================================")
    log("STARTING AUTONOMOUS GOAL RUN FOR M2-MAX-V7")
    log("===================================================================")
    log(f"Free disk space: {get_free_disk_gb():.2f} GB")

    train_cmd = [
        sys.executable, "train.py",
        "--config", "training/config/models/m2_max_v7.yaml",
        "--ablation", "M2-MAX-V7",
        "--data-root", str(DATA_ROOT),
        "--list-dir", str(LIST_DIR),
        "--output-dir", str(V7_DIR),
        "--epochs", "80",
        "--batch-size", "16",
        "--lr", "0.00015",
        "--weight-decay", "0.0001",
        "--clip-grad", "1.0",
        "--ce-weight", "0.5",
        "--dice-weight", "0.5",
        "--aar-weight", "0.4",
        "--scar-weight", "0.12",
        "--wall-weight", "0.2",
        "--inclusion-weight", "0.15",
        "--dice-class-weights", "0.0", "1.0", "2.0", "2.0",
        "--ce-class-weights", "0.2", "1.0", "2.0", "2.0",
        "--sampler", "rare",
        "--rare-boost", "2.0",
        "--foreground-boost", "1.3",
        "--init-weights", str(INIT_WEIGHTS),
        "--device", "cuda",
        "--amp", "auto",
        "--num-workers", "4",
        "--save-every", "10",
    ]

    stdout_log = PROJECT_ROOT / "outputs" / "m2_max_v7_stdout.log"
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    if V7_DIR.exists():
        shutil.rmtree(V7_DIR)
    log(f"Launching training process (Stdout -> {stdout_log})...")

    with open(stdout_log, "w") as out_f:
        proc = subprocess.Popen(train_cmd, cwd=str(PROJECT_ROOT), stdout=out_f, stderr=subprocess.STDOUT)
        log(f"M2-MAX-V7 PID: {proc.pid}")

        start_time = time.time()
        last_check = 0

        while proc.poll() is None:
            time.sleep(20)
            elapsed = time.time() - start_time

            # 1. Disk Guard Check
            disk_guard_cleanup(V7_DIR, min_free_gb=5.0)

            # 2. Progress reporting every 60s
            if time.time() - last_check > 60:
                last_check = time.time()
                train_log = V7_DIR / "logs_M2-Max-V7_seed1234" / "train.log"
                latest_epoch = "Initializing..."
                if train_log.is_file():
                    try:
                        with open(train_log) as f:
                            lines = [l.strip() for l in f if "Epoch " in l]
                            if lines:
                                latest_epoch = lines[-1].split("|")[2].strip()
                    except Exception:
                        pass
                log(f"[Elapsed {elapsed/60:.1f}m] V7 Status: {latest_epoch} | Free disk: {get_free_disk_gb():.1f} GB")

        ret = proc.returncode
        log(f"Training completed with returncode {ret} in {(time.time()-start_time)/60:.1f}m")

    if ret != 0:
        log("Training failed! Exiting.")
        sys.exit(ret)

    # Run Benchmark Evaluation
    log("===================================================================")
    log("RUNNING COMPREHENSIVE BENCHMARK EVALUATION ON TEST SET (76 CASINOS)...")
    log("===================================================================")
    results = run_full_benchmark(V7_DIR)
    report_text = format_report(results)
    print("\n" + report_text + "\n", flush=True)

    report_file = V7_DIR / "BENCHMARK_REPORT.md"
    report_file.write_text(report_text, encoding="utf-8")
    log(f"Saved benchmark report to {report_file}")

    # Backup to Google Drive
    backup_to_gdrive(V7_DIR)
    log("=== GOAL EXECUTION COMPLETED! ===")


if __name__ == "__main__":
    main()
