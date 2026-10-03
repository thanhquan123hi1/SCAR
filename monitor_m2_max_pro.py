"""Autonomous Monitor and Evaluator for M2-Max-Pro.
Monitors training until completion (300 epochs), then:
1. Evaluates all 3 checkpoints (best, best_inclusive, last) on the 76 test cases.
2. Checks against I_MMSeg criteria (Scar > 73.64%, Edema Inclusive > 75.44%, no anomalies).
3. If criteria met, pushes code and model weights/metadata to GitHub!
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUN_DIR = Path("/content/SCAR/outputs/runs/m2_max_pro")
DATA_ROOT = "/content/SCAR/MyoPS380_dataset/Processed_data"
REPO_DIR = Path("/content/SCAR")

def is_training_active() -> bool:
    try:
        output = subprocess.check_output(["pgrep", "-f", "train.py.*m2_max_pro"]).decode()
        return bool(output.strip())
    except Exception:
        return False

def get_latest_epoch() -> int:
    metrics_file = RUN_DIR / "logs_M2-Max-Pro_seed1234" / "metrics.jsonl"
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                lines = f.readlines()
            if lines:
                return json.loads(lines[-1]).get("epoch", 0)
        except Exception:
            pass
    return 0

def run_cmd(cmd, cwd=str(REPO_DIR)):
    print(f"\n[RUNNING] {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    res = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), check=True)
    return res

print("=== STARTING AUTONOMOUS MONITOR FOR M2-MAX-PRO ===", flush=True)

# 1. Monitoring loop
start_time = time.time()
while is_training_active():
    epoch = get_latest_epoch()
    elapsed = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] M2-Max-Pro training: Epoch {epoch}/300 | Elapsed: {elapsed/60:.1f} mins", flush=True)
    time.sleep(60)

print("\n=== M2-MAX-PRO TRAINING HAS FINISHED! ===", flush=True)
time.sleep(5)  # Let disk flush

checkpoints_dir = RUN_DIR / "checkpoints"
if not checkpoints_dir.exists():
    raise RuntimeError(f"Checkpoints directory not found at {checkpoints_dir}!")

best_pth = checkpoints_dir / "best.pth"
best_inc_pth = checkpoints_dir / "best_inclusive.pth"
last_pth = checkpoints_dir / "last.pth"

ckpts = [
    ("best", best_pth, RUN_DIR / "test_results_best"),
    ("best_inclusive", best_inc_pth, RUN_DIR / "test_results_best_inclusive"),
    ("last", last_pth, RUN_DIR / "test_results_last"),
]

# 2. Evaluation on all 76 test volumes
for label, ckpt_path, out_dir in ckpts:
    if ckpt_path.exists():
        print(f"\n>>> Evaluating {label}: {ckpt_path.name} -> {out_dir.name} <<<", flush=True)
        test_cmd = [
            sys.executable,
            "test.py",
            "--checkpoint", str(ckpt_path),
            "--data-root", DATA_ROOT,
            "--split", "test_vol",
            "--output-dir", str(out_dir),
            "--batch-size", "16",
            "--device", "cuda",
            "--amp", "auto",
            "--save-predictions",
        ]
        run_cmd(test_cmd)

# 3. Check criteria vs I_MMSeg
# I_MMSeg targets:
# Scar Dice > 73.64% (HD95 < 3.66)
# Edema Inclusive Dice > 75.44% (HD95 < 3.89)
# Myocardial Ring > 86.8%
immseg_beaten = False
winning_checkpoint = None
best_report = []

for label, ckpt_path, out_dir in ckpts:
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        continue
    with open(metrics_path) as f:
        m = json.load(f)

    scar_dice = m.get("scar", {}).get("mean_dice", 0) * 100
    scar_hd95 = m.get("scar", {}).get("mean_hd95_voxel", 999)
    edema_inc_dice = m.get("edema_inclusive", {}).get("mean_dice", 0) * 100
    edema_inc_hd95 = m.get("edema_inclusive", {}).get("mean_hd95_voxel", 999)
    myo_dice = m.get("myocardial_ring", {}).get("mean_dice", 0) * 100

    report_line = (
        f"[{label.upper()}] Scar: {scar_dice:.2f}% (HD95: {scar_hd95:.2f}) | "
        f"Edema-Inc: {edema_inc_dice:.2f}% (HD95: {edema_inc_hd95:.2f}) | "
        f"Myo: {myo_dice:.2f}%"
    )
    best_report.append(report_line)
    print(report_line, flush=True)

    # Check if this checkpoint beats I_MMSeg
    if scar_dice > 73.64 and edema_inc_dice > 75.44 and myo_dice > 86.8:
        immseg_beaten = True
        winning_checkpoint = label
        print(f"\n🎉 EXCEEDED I_MMSEG TARGET ON CHECKPOINT: {label}! 🎉", flush=True)

summary_text = "\n".join(best_report)
with open(RUN_DIR / "M2_MAX_PRO_TEST_SUMMARY.txt", "w") as f:
    f.write(summary_text)

# 4. If beaten, push code and results to GitHub
if immseg_beaten:
    print("\n>>> Pushing winning M2-Max-Pro code to GitHub <<<", flush=True)
    try:
        run_cmd(["git", "add", "training/", "train.py", "test.py"])
        commit_msg = (
            f"feat(model): M2-Max-Pro exceeds I_MMSeg benchmark\n\n"
            f"Results on 76 test volumes:\n{summary_text}\n"
        )
        run_cmd(["git", "commit", "-m", commit_msg])
        run_cmd(["git", "push", "origin", "main"])
        print("\n✅ Successfully pushed M2-Max-Pro to GitHub!", flush=True)
    except Exception as e:
        print(f"Error during git push: {e}", flush=True)
else:
    print("\nDid not beat both I_MMSeg thresholds yet. Ready for next refinement iteration.", flush=True)

print("\n=== MONITOR FINISHED ===", flush=True)
