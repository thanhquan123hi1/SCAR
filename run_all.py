"""1-Click End-to-End Pipeline for LGE Cardiac MRI Segmentation.

Usage:
    python run_all.py --config training/config/models/unet_3d.yaml --run-id unet3d_exp01
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("1-Click-Runner")


def run_command(cmd: list[str], desc: str) -> None:
    logger.info(">>> BƯỚC: %s", desc)
    logger.info("Thực thi: %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        logger.error("Lỗi khi thực thi lệnh (mã lỗi %d): %s", res.returncode, " ".join(cmd))
        sys.exit(res.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="1-Click LGE Training & Evaluation Pipeline.")
    parser.add_argument("--config", default="training/config/models/unet_3d.yaml", help="Model config YAML")
    parser.add_argument("--run-id", default="unet3d_lge_sax_run01", help="Unique run identifier")
    parser.add_argument("--data-root", default="data/LGE_MULTI", help="LGE dataset root")
    parser.add_argument("--skip-cache", action="store_true", help="Skip offline preprocessing caching")
    parser.add_argument("--eval-split", default="validation", choices=["train", "validation", "test"])
    args = parser.parse_args()

    python_bin = sys.executable

    # 1. Build splits if missing
    splits_dir = ROOT / "data/processed/splits"
    if not (splits_dir / "train.csv").exists():
        run_command(
            [
                python_bin,
                "preprocessing/build_splits.py",
                "--data-root",
                args.data_root,
                "--output",
                "data/processed/splits",
            ],
            "1. Sinh danh sách splits train/val/test CSV",
        )
    else:
        logger.info("Splits CSV đã tồn tại tại %s. Bỏ qua bước 1.", splits_dir)

    # 2. Offline caching
    cache_dir = ROOT / "data/processed/cache"
    if not args.skip_cache:
        run_command(
            [
                python_bin,
                "preprocessing/process_and_save.py",
                "--data-root",
                args.data_root,
                "--splits-dir",
                "data/processed/splits",
                "--output-dir",
                "data/processed/cache",
            ],
            "2. Tiền xử lý & Caching dữ liệu (.npz) để nạp siêu tốc",
        )

    # 3. Train model
    run_command(
        [
            python_bin,
            "training/train.py",
            "--config",
            args.config,
            "--run-id",
            args.run_id,
        ],
        f"3. Huấn luyện mô hình LGE ({args.run_id})",
    )

    # 4. Evaluate & scar quantification
    run_dir = ROOT / "outputs/runs" / args.run_id
    config_snapshot = run_dir / "config_snapshot.yaml"
    best_checkpoint = run_dir / "checkpoints/best.pt"

    if best_checkpoint.exists():
        run_command(
            [
                python_bin,
                "training/evaluate.py",
                "--config",
                str(config_snapshot),
                "--checkpoint",
                str(best_checkpoint),
                "--split",
                args.eval_split,
            ],
            f"4. Đánh giá chi tiết & Định lượng sẹo cơ tim trên tập {args.eval_split}",
        )

    logger.info("🎉 TOÀN BỘ QUY TRÌNH 1-CLICK ĐÃ HOÀN TẤT THÀNH CÔNG!")
    logger.info("Checkpoints và Báo cáo đã lưu tại: %s", run_dir)


if __name__ == "__main__":
    main()
