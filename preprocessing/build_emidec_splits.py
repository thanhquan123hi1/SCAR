"""Scan EMIDEC dataset directory and build train/val/test CSV splits.

Reference:
    - EMIDEC Challenge (Automatic Evaluation of Myocardial Infarction from DE-MRI).
    - Structure: Case_Nxxx, Case_Pxxx containing MRI images (in Images/ or root) and Contours (in Contours/).

Usage:
    python preprocessing/build_emidec_splits.py \
        --data-root data/LGE_MULTI \
        --output data/processed/splits \
        --val-ratio 0.2 \
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("BuildEmidecSplits")


def find_case_files(case_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Locate the 3D MRI image and Contour/Ground-truth NIfTI files inside an EMIDEC Case folder."""
    all_nii = sorted(list(case_dir.rglob("*.nii*")))
    if not all_nii:
        return None, None

    img_path = None
    lbl_path = None

    for p in all_nii:
        p_str = str(p).lower()
        p_parent = p.parent.name.lower()
        p_name = p.name.lower()

        # Check if file belongs to Contours
        if "contour" in p_parent or "contour" in p_name or "groundtruth" in p_name or "mask" in p_name:
            lbl_path = p
        elif "image" in p_parent:
            img_path = p
        elif p_parent.startswith("case_") and img_path is None:
            img_path = p

    # Fallback if only 2 files exist
    if (img_path is None or lbl_path is None) and len(all_nii) >= 2:
        for p in all_nii:
            if "contour" in str(p).lower():
                lbl_path = p
            else:
                img_path = p

    if img_path is None and len(all_nii) > 0:
        img_path = all_nii[0]

    return img_path, lbl_path


def scan_emidec(data_root: Path) -> List[Dict]:
    """Scan all Case directories in data_root."""
    if not data_root.exists():
        raise FileNotFoundError(f"Thư mục data_root không tồn tại: {data_root}")

    # Look for Case_* folders directly or inside subfolders
    case_dirs = [p for p in data_root.iterdir() if p.is_dir() and p.name.lower().startswith("case_")]

    if not case_dirs:
        for sub in data_root.rglob("Case_*"):
            if sub.is_dir():
                case_dirs.append(sub)

    case_dirs = sorted(list(set(case_dirs)), key=lambda p: p.name)
    logger.info("Found %d case folders in %s", len(case_dirs), data_root)

    records = []
    labels_found = 0
    for c_dir in case_dirs:
        case_name = c_dir.name
        img_path, lbl_path = find_case_files(c_dir)

        if img_path is None:
            logger.warning("No image found in %s, skipping.", c_dir)
            continue

        if lbl_path is not None:
            labels_found += 1

        rel_img = img_path.relative_to(data_root).as_posix()
        rel_lbl = lbl_path.relative_to(data_root).as_posix() if lbl_path else None

        records.append({
            "record_id": f"emidec_{case_name.lower()}",
            "view": "SAX",
            "split": "train",
            "subject_id": case_name,
            "image_path": rel_img,
            "label_path": rel_lbl,
            "has_label": rel_lbl is not None,
        })

    logger.info("Summary: Total cases: %d | Ground-truth masks found: %d", len(records), labels_found)
    return records


def split_records(
    records: List[Dict],
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> List[Dict]:
    """Split records into train, validation, and test subsets."""
    random.seed(seed)
    indices = list(range(len(records)))
    random.shuffle(indices)

    n_total = len(records)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_val - n_test

    for i, idx in enumerate(indices):
        if i < n_train:
            records[idx]["split"] = "train"
        elif i < n_train + n_val:
            records[idx]["split"] = "validation"
        else:
            records[idx]["split"] = "test"

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EMIDEC train/val/test splits.")
    parser.add_argument("--data-root", default="data/LGE_MULTI", help="Path to EMIDEC root")
    parser.add_argument("--output", default="data/processed/splits", help="Output directory for CSV files")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (default: 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Test ratio (default: 0.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    logger.info("Scanning EMIDEC data root: %s", data_root)

    records = scan_emidec(data_root)
    if not records:
        raise RuntimeError(f"Không tìm thấy ca bệnh nào trong '{data_root}'.")

    records = split_records(records, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)

    df = pd.DataFrame(records)
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "manifest.csv", index=False)

    for split in ("train", "validation", "test"):
        sub = df[df["split"] == split]
        if len(sub) > 0:
            sub.to_csv(out_dir / f"{split}.csv", index=False)
            logger.info("  %s: %d records", f"{split:12s}", len(sub))

    logger.info("EMIDEC Splits CSVs successfully saved to: %s", out_dir)


if __name__ == "__main__":
    main()
