"""Scan LGE_MULTI directory and build train/val/test CSV splits.

Usage:
    python preprocessing/build_splits.py \
        --data-root data/LGE_MULTI \
        --output data/processed/splits
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("BuildSplits")

TASK_DIR_PATTERN = re.compile(r"^(SAX|2CH|4CH|RAS)_(TR|VAL|TST)$", re.IGNORECASE)
FILE_PATTERN = re.compile(r"^(?:LGE_)?(SAX|2CH|4CH|RAS)_(\d+)\.nii(?:\.gz)?$", re.IGNORECASE)
SPLIT_ALIASES = {"TR": "train", "VAL": "validation", "TST": "test"}


def get_lge_directory(data_root: Path) -> Path:
    """Find the directory that directly contains task folders like SAX_TR."""
    if not data_root.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {data_root}")

    # 1. Check if data_root itself contains SAX_TR / SAX_VAL
    direct_match = any(TASK_DIR_PATTERN.match(p.name) for p in data_root.iterdir() if p.is_dir())
    if direct_match:
        return data_root

    # 2. Check if data_root/LGE_MULTI exists
    if (data_root / "LGE_MULTI").exists():
        return data_root / "LGE_MULTI"

    # 3. Look for any directory named LGE_MULTI inside data_root
    for p in data_root.rglob("LGE_MULTI"):
        if p.is_dir():
            return p

    # 4. Search recursively for any folder containing task dirs (ignoring CINE)
    for p in data_root.rglob("*_TR"):
        if p.is_dir() and "CINE" not in p.as_posix().upper():
            return p.parent

    raise FileNotFoundError(
        f"Không tìm thấy các thư mục LGE (như SAX_TR, SAX_VAL) trong '{data_root}'.\n"
        f"Nội dung hiện tại trong '{data_root}': {[p.name for p in data_root.iterdir()]}"
    )


def scan_lge(lge_dir: Path) -> list[dict]:
    records = []
    for task_dir in sorted(lge_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        match = TASK_DIR_PATTERN.match(task_dir.name)
        if not match:
            continue
        view, split_code = match.groups()
        view = view.upper()
        split = SPLIT_ALIASES.get(split_code.upper(), split_code.lower())

        image_dir = task_dir / "image"
        label_dir = task_dir / "anno"
        if not image_dir.exists():
            continue

        label_map = (
            {p.name: p for p in label_dir.glob("*.nii*")}
            if label_dir.exists()
            else {}
        )

        for img_path in sorted(image_dir.glob("*.nii*")):
            fm = FILE_PATTERN.match(img_path.name)
            if not fm:
                continue
            subject_id = f"{int(fm.group(2)):03d}"
            label_path = label_map.get(img_path.name)

            records.append(
                {
                    "record_id": f"lge_{view.lower()}_{split}_{subject_id}",
                    "view": view,
                    "split": split,
                    "subject_id": subject_id,
                    "image_path": img_path.relative_to(lge_dir).as_posix(),
                    "label_path": (
                        label_path.relative_to(lge_dir).as_posix()
                        if label_path
                        else None
                    ),
                    "has_label": label_path is not None,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LGE train/val/test splits.")
    parser.add_argument("--data-root", default="data/LGE_MULTI", help="Path to LGE_MULTI data directory")
    parser.add_argument("--output", default="data/processed/splits", help="Output directory for CSV files")
    parser.add_argument("--view", default=None, help="Filter by view: SAX, 2CH, 4CH, RAS")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    logger.info("Scanning data root: %s", data_root)

    lge_dir = get_lge_directory(data_root)
    logger.info("Found LGE task folders at: %s", lge_dir)

    records = scan_lge(lge_dir)
    if not records:
        raise RuntimeError(
            f"Không tìm thấy file ảnh LGE nào trong '{lge_dir}'. "
            f"Hãy đảm bảo cấu trúc gồm SAX_TR/image/*.nii.gz, SAX_VAL/image/*.nii.gz..."
        )

    if args.view:
        records = [r for r in records if r["view"] == args.view.upper()]

    df = pd.DataFrame(records)
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest
    df.to_csv(out_dir / "manifest.csv", index=False)

    # Save per-split CSVs
    for split in ("train", "validation", "test"):
        subset = df[df["split"] == split]
        subset.to_csv(out_dir / f"{split}.csv", index=False)
        logger.info("  %s: %d records", f"{split:12s}", len(subset))

    logger.info("Splits CSVs saved to %s", out_dir)
    logger.info("Total: %d records across %d views.", len(df), df["view"].nunique())

    # ---- Patient-level data leakage check ----
    for view in df["view"].unique():
        view_df = df[df["view"] == view]
        splits_present = view_df["split"].unique()
        if len(splits_present) <= 1:
            continue
        for s1 in splits_present:
            for s2 in splits_present:
                if s1 >= s2:
                    continue
                ids_s1 = set(view_df[view_df["split"] == s1]["subject_id"])
                ids_s2 = set(view_df[view_df["split"] == s2]["subject_id"])
                overlap = ids_s1 & ids_s2
                if overlap:
                    logger.warning(
                        "⚠️ DATA LEAKAGE: View '%s' has %d overlapping patients between '%s' and '%s': %s",
                        view, len(overlap), s1, s2, overlap,
                    )
                else:
                    logger.info("  ✓ View '%s': '%s' ∩ '%s' = ∅ (no leakage)", view, s1, s2)


if __name__ == "__main__":
    main()
