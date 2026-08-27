"""Test end-to-end evaluation pipeline and summary metrics generation."""

import tempfile
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.evaluate import evaluate_split


class DummyModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # x is (B, 1, 16, 192, 192) or (B, 1, 256, 256)
        shape = (x.shape[0], self.num_classes) + tuple(x.shape[2:])
        out = torch.zeros(shape, device=x.device, dtype=torch.float32)
        # Predict class 2 (myocardium) in center
        out[:, 2, ...] = 2.0
        return out


def test_evaluate_split_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_root = tmp_path / "data"
        data_root.mkdir()
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        # Create dummy 3D SAX image and label NIfTI files
        affine = np.diag([1.0, 1.0, 10.0, 1.0])

        # Subject 1: has myocardium and scar
        img1 = (np.random.rand(192, 192, 16) * 100).astype(np.float32)
        lbl1 = np.zeros((192, 192, 16), dtype=np.int16)
        lbl1[80:120, 80:120, 4:12] = 2  # Myo
        lbl1[90:100, 90:100, 6:8] = 3   # Scar

        nib.save(nib.Nifti1Image(img1, affine), str(data_root / "sub1_img.nii.gz"))
        nib.save(nib.Nifti1Image(lbl1, affine), str(data_root / "sub1_lbl.nii.gz"))

        # Subject 2: True Negative case (healthy, no scar)
        img2 = (np.random.rand(192, 192, 16) * 100).astype(np.float32)
        lbl2 = np.zeros((192, 192, 16), dtype=np.int16)
        lbl2[80:120, 80:120, 4:12] = 2  # Myo only, NO scar

        nib.save(nib.Nifti1Image(img2, affine), str(data_root / "sub2_img.nii.gz"))
        nib.save(nib.Nifti1Image(lbl2, affine), str(data_root / "sub2_lbl.nii.gz"))

        manifest = pd.DataFrame([
            {
                "subject_id": "001",
                "record_id": "rec_001",
                "image_path": "sub1_img.nii.gz",
                "label_path": "sub1_lbl.nii.gz",
                "has_label": True,
                "view": "SAX",
            },
            {
                "subject_id": "002",
                "record_id": "rec_002",
                "image_path": "sub2_img.nii.gz",
                "label_path": "sub2_lbl.nii.gz",
                "has_label": True,
                "view": "SAX",
            },
        ])

        config = {
            "num_classes": 5,
            "view": "SAX",
            "preprocessing": {
                "target_shape": [192, 192, 16],
                "target_spacing": [1.0, 1.0, 10.0],
                "intensity_percentiles": [1.0, 99.0],
            },
            "postprocess": {
                "use_rules": False,
                "anatomical_constraint": True,
                "tolerance_mm": 2.5,
                "min_scar_volume_mm3": 15.0,
            },
        }

        model = DummyModel(num_classes=5)
        device = torch.device("cpu")

        subj_df, summ_df = evaluate_split(
            model=model,
            df=manifest,
            data_root=data_root,
            config=config,
            device=device,
            save_predictions=True,
            output_dir=out_dir,
        )

        assert len(subj_df) == 2, "Expected 2 evaluated subjects"
        assert len(summ_df) > 0, "Summary dataframe must have rows"
        
        # Verify columns in subject dataframe
        assert "dice_scar" in subj_df.columns
        assert "dice_conditional_scar" in subj_df.columns
        assert "hd95_scar_mm" in subj_df.columns
        assert "hd95_conditional_scar_mm" in subj_df.columns
        assert "fov_diagonal_mm" in subj_df.columns

        # Verify subject 2 (True Negative for scar: GT=0, Pred=0)
        row_tn = subj_df.iloc[1]
        assert row_tn["has_gt_scar"] is False or row_tn["has_gt_scar"] == 0
        assert row_tn["dice_scar"] == 1.0, "TN Overall Dice must be 1.0"
        assert row_tn["hd95_scar_mm"] == 0.0, "TN Overall HD95 must be 0.0 mm"
        assert np.isnan(row_tn["dice_conditional_scar"]), "TN Conditional Dice must be NaN"
        assert np.isnan(row_tn["hd95_conditional_scar_mm"]), "TN Conditional HD95 must be NaN"

        # Verify summary dataframe columns: median, iqr, mean, std
        for req_col in ["metric", "mean", "std", "median", "iqr", "q25", "q75", "count"]:
            assert req_col in summ_df.columns, f"Missing {req_col} in summary table"

        # Verify exported files exist
        assert (out_dir / "per_subject_metrics.csv").exists()
        assert (out_dir / "per_class_summary.csv").exists()
        assert (out_dir / "nifti_predictions" / "rec_001_pred.nii.gz").exists()
        assert (out_dir / "nifti_predictions" / "rec_002_pred.nii.gz").exists()

        print("  -> End-to-End Evaluation Pipeline: VERIFIED")


if __name__ == "__main__":
    test_evaluate_split_end_to_end()
    print("All evaluate pipeline tests passed!")
