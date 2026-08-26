"""Quick sanity check: load a few LGE NIfTI files and run preprocessing.

Usage:
    python preprocessing/verify.py \
        --data-root data/LGE_MULTI \
        --config preprocessing/config.yaml \
        --n 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.preprocessing import preprocess_mask, preprocess_spatial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/LGE_MULTI")
    parser.add_argument("--config", default="preprocessing/config.yaml")
    parser.add_argument("--view", default="SAX", choices=["SAX", "2CH", "4CH", "RAS"])
    parser.add_argument("--n", type=int, default=3, help="Number of files to verify")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    view_key = "sax" if args.view == "SAX" else "lax"
    pp = cfg["lge_preprocessing"][view_key]

    target_shape = tuple(pp["target_shape"])
    target_spacing = tuple(pp["target_spacing"])
    percentiles = tuple(pp["intensity_percentiles"]) if pp["intensity_percentiles"] else None

    data_root = Path(args.data_root).resolve()
    image_files = sorted(data_root.rglob(f"*{args.view}_TR/image/*.nii*"))[: args.n]

    if not image_files:
        print(f"Không tìm thấy file ảnh nào trong {data_root} cho view {args.view}")
        return

    for img_path in image_files:
        nii = nib.load(str(img_path))
        image = np.asanyarray(nii.dataobj)
        spacing = tuple(float(v) for v in nii.header.get_zooms()[: len(target_shape)])

        processed, transform = preprocess_spatial(
            image,
            source_spacing=spacing,
            target_spacing=target_spacing,
            target_shape=target_shape,
            interpolation_order=1,
            intensity_percentiles=percentiles,
        )
        print(
            f"  {img_path.name}: {image.shape} {spacing} mm"
            f"  →  {processed.shape}  min={processed.min():.3f}  max={processed.max():.3f}"
        )

        # Check label if available
        label_path = img_path.parent.parent / "anno" / img_path.name
        if label_path.exists():
            label_nii = nib.load(str(label_path))
            label = np.rint(np.asanyarray(label_nii.dataobj)).astype(np.int16)
            label_proc = preprocess_mask(
                label,
                source_spacing=spacing,
                target_spacing=target_spacing,
                target_shape=target_shape,
            )
            print(f"    label unique values: {sorted(np.unique(label_proc).tolist())}")

    print("\nSanity check PASSED.")


if __name__ == "__main__":
    main()
