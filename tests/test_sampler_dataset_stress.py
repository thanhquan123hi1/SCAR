"""Adversarial stress testing for Sampler and Dataset Integration (M3).

Tests:
1. Sampler empty / single / uniform / extreme distribution edge cases.
2. Sampler returning None on uniform weights enabling standard DataLoader shuffle.
3. Sampler empirical distribution matching theoretical boost ratios.
4. Sampler error handling / corrupted file resilience (no crashes, default weight 1.0).
5. Strict mode vs non-strict mode behaviors.
6. LgeLaxDataset in_channels=1 vs in_channels=3 boundary clamping logic (cached & on-the-fly).
7. Single-slice (D=1) volume boundary handling in 2.5D.
8. Integration between LgeLaxDataset, build_rare_class_sampler, and PyTorch DataLoader.
"""

from __future__ import annotations

import logging
import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from training.dataset.lge_dataset import LgeLaxDataset, LgeSaxDataset
from training.dataset.sampler import build_rare_class_sampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sampler_stress_test")


class DummyDataset(Dataset):
    """Minimal dataset mock for testing raw sampler behaviors."""
    def __init__(self, records: pd.DataFrame, slices: list[dict] | None = None, cache_dir: Path | None = None):
        self.records = records
        if slices is not None:
            self.slices = slices
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        if hasattr(self, "slices"):
            return len(self.slices)
        return len(self.records)

    def __getitem__(self, idx: int):
        return {"index": idx}


def test_sampler_empty_and_single_dataset():
    """Test sampler with empty and single sample datasets."""
    logger.info("Running test_sampler_empty_and_single_dataset...")
    # 1. Empty dataset
    empty_df = pd.DataFrame(columns=["record_id", "subject_id", "has_label", "label_path"])
    ds_empty = DummyDataset(records=empty_df)
    sampler_empty = build_rare_class_sampler(ds_empty)
    assert sampler_empty is None, "Empty dataset must return None"

    # 2. Single sample dataset (uniform weight -> returns None)
    single_df = pd.DataFrame([{"record_id": "rec_0", "subject_id": "sub_0", "has_label": False}])
    ds_single = DummyDataset(records=single_df)
    sampler_single = build_rare_class_sampler(ds_single, strict=False)
    assert sampler_single is None, "Single sample dataset has uniform weight and must return None in non-strict mode"

    # Single sample strict mode (default weight=1.0 raises RuntimeError)
    try:
        build_rare_class_sampler(ds_single, strict=True)
        assert False, "Strict mode on missing cache with weight=1.0 must raise RuntimeError"
    except RuntimeError as e:
        assert "All 1 samples have default weight=1.0" in str(e)


def test_sampler_uniform_weights_scenarios():
    """Test all scenarios where sampler weights are uniform."""
    logger.info("Running test_sampler_uniform_weights_scenarios...")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # Scenario A: All pure background volumes (labels exist, but all 0)
        recs = []
        for i in range(5):
            rec_id = f"bg_{i}"
            np.savez(tmp_dir / f"{rec_id}.npz", image=np.zeros((10, 32, 32)), label=np.zeros((10, 32, 32), dtype=np.int16))
            recs.append({"record_id": rec_id, "subject_id": f"sub_{i}", "has_label": True})
        df_bg = pd.DataFrame(recs)
        ds_bg = DummyDataset(records=df_bg, cache_dir=tmp_dir)
        sampler_bg = build_rare_class_sampler(ds_bg)
        assert sampler_bg is None, "All pure-background dataset must return None"

        # Scenario B: All pure rare class volumes (all samples contain scar class 3)
        recs_rare = []
        for i in range(5):
            rec_id = f"rare_{i}"
            lbl = np.zeros((10, 32, 32), dtype=np.int16)
            lbl[2, 5:10, 5:10] = 3
            np.savez(tmp_dir / f"{rec_id}.npz", image=np.zeros((10, 32, 32)), label=lbl)
            recs_rare.append({"record_id": rec_id, "subject_id": f"sub_{i}", "has_label": True})
        df_rare = pd.DataFrame(recs_rare)
        ds_rare = DummyDataset(records=df_rare, cache_dir=tmp_dir)
        sampler_rare = build_rare_class_sampler(ds_rare)
        assert sampler_rare is None, "All pure-rare dataset (all weights=2.0) must return None"

        # Scenario C: All foreground class 2 (myo) volumes (all weights=1.3)
        recs_myo = []
        for i in range(5):
            rec_id = f"myo_{i}"
            lbl = np.zeros((10, 32, 32), dtype=np.int16)
            lbl[2, 5:10, 5:10] = 2
            np.savez(tmp_dir / f"{rec_id}.npz", image=np.zeros((10, 32, 32)), label=lbl)
            recs_myo.append({"record_id": rec_id, "subject_id": f"sub_{i}", "has_label": True})
        df_myo = pd.DataFrame(recs_myo)
        ds_myo = DummyDataset(records=df_myo, cache_dir=tmp_dir)
        sampler_myo = build_rare_class_sampler(ds_myo)
        assert sampler_myo is None, "All foreground-only dataset (all weights=1.3) must return None"

        # Verify fallback with DataLoader
        loader = DataLoader(ds_bg, batch_size=2, shuffle=(sampler_bg is None))
        batches = list(loader)
        assert len(batches) == 3, f"Expected 3 batches for 5 items with batch_size=2, got {len(batches)}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_sampler_empirical_draw_distribution():
    """Adversarially verify weighted random sampling empirical probabilities."""
    logger.info("Running test_sampler_empirical_draw_distribution...")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # Create dataset: 10 background samples (weight 1.0), 10 rare scar samples (weight 2.0)
        recs = []
        for i in range(10):
            rec_id = f"bg_{i}"
            np.savez(tmp_dir / f"{rec_id}.npz", image=np.zeros((32, 32)), label=np.zeros((32, 32), dtype=np.int16))
            recs.append({"record_id": rec_id, "subject_id": f"sub_{i}", "has_label": True})
        for i in range(10):
            rec_id = f"rare_{i}"
            lbl = np.zeros((32, 32), dtype=np.int16)
            lbl[5:10, 5:10] = 3
            np.savez(tmp_dir / f"{rec_id}.npz", image=np.zeros((32, 32)), label=lbl)
            recs.append({"record_id": rec_id, "subject_id": f"sub_{i+10}", "has_label": True})

        df = pd.DataFrame(recs)
        ds = DummyDataset(records=df, cache_dir=tmp_dir)
        rare_boost = 3.0
        sampler = build_rare_class_sampler(ds, rare_boost=rare_boost, foreground_boost=1.5)
        assert sampler is not None, "Sampler must not be None when weights are non-uniform"
        assert len(sampler.weights) == 20
        assert (sampler.weights[:10] == 1.0).all()
        assert (sampler.weights[10:] == 3.0).all()

        # Empirical simulation: draw 50,000 samples and measure relative frequencies
        num_draws = 50000
        # Draw indices using sampler weights
        drawn_indices = torch.multinomial(sampler.weights, num_draws, replacement=True).numpy()
        bg_draws = (drawn_indices < 10).sum()
        rare_draws = (drawn_indices >= 10).sum()

        # Theoretical ratio: (10 * 3.0) / (10 * 1.0) = 3.0
        observed_ratio = rare_draws / bg_draws
        expected_ratio = rare_boost / 1.0
        logger.info("Empirical sampling test: observed ratio = %.4f, expected = %.4f", observed_ratio, expected_ratio)
        # Margin of error for 50,000 draws should be within 5%
        assert math.isclose(observed_ratio, expected_ratio, rel_tol=0.06), (
            f"Empirical ratio {observed_ratio:.4f} deviated from expected {expected_ratio:.4f}"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_sampler_extreme_class_distribution():
    """Test extreme distributions: 1 rare vs 999 background samples."""
    logger.info("Running test_sampler_extreme_class_distribution...")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        recs = []
        # 999 background
        for i in range(999):
            rec_id = f"bg_{i}"
            np.savez(tmp_dir / f"{rec_id}.npz", image=np.zeros((8, 8)), label=np.zeros((8, 8), dtype=np.int16))
            recs.append({"record_id": rec_id, "subject_id": f"sub_{i}", "has_label": True})
        # 1 rare scar
        rec_id = "rare_0"
        lbl = np.zeros((8, 8), dtype=np.int16)
        lbl[2, 2] = 3
        np.savez(tmp_dir / f"{rec_id}.npz", image=np.zeros((8, 8)), label=lbl)
        recs.append({"record_id": rec_id, "subject_id": "sub_rare", "has_label": True})

        df = pd.DataFrame(recs)
        ds = DummyDataset(records=df, cache_dir=tmp_dir)
        sampler = build_rare_class_sampler(ds, rare_boost=5.0)
        assert sampler is not None
        assert len(sampler.weights) == 1000
        assert sampler.weights[-1] == 5.0
        assert (sampler.weights[:-1] == 1.0).all()

        # Check PyTorch DataLoader instantiation and iteration
        loader = DataLoader(ds, batch_size=32, sampler=sampler)
        batches = list(loader)
        assert len(batches) == math.ceil(1000 / 32)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_sampler_corrupted_and_missing_file_resilience():
    """Test sampler resilience when files are corrupted, missing, or have invalid formats."""
    logger.info("Running test_sampler_corrupted_and_missing_file_resilience...")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # File 1: corrupted npz (junk bytes)
        (tmp_dir / "rec_corrupt.npz").write_bytes(b"NOT_A_VALID_ZIP_ARCHIVE")

        # File 2: valid npz with rare scar
        lbl = np.zeros((8, 8), dtype=np.int16)
        lbl[1, 1] = 3
        np.savez(tmp_dir / "rec_valid.npz", image=np.zeros((8, 8)), label=lbl)

        # File 3: missing npz
        df = pd.DataFrame([
            {"record_id": "rec_corrupt", "subject_id": "s1", "has_label": True},
            {"record_id": "rec_valid", "subject_id": "s2", "has_label": True},
            {"record_id": "rec_missing", "subject_id": "s3", "has_label": True},
        ])
        ds = DummyDataset(records=df, cache_dir=tmp_dir)
        # Should not crash, rec_corrupt and rec_missing should get default weight=1.0, rec_valid gets weight=2.0
        sampler = build_rare_class_sampler(ds, rare_boost=2.0)
        assert sampler is not None
        assert sampler.weights[0] == 1.0  # Corrupt -> fallback 1.0
        assert sampler.weights[1] == 2.0  # Valid rare -> 2.0
        assert sampler.weights[2] == 1.0  # Missing -> 1.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_dataset_2d_vs_25d_boundary_clamping():
    """Adversarially verify 2.5D boundary slice clamping logic vs 2D slice extraction."""
    logger.info("Running test_dataset_2d_vs_25d_boundary_clamping...")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # Create a 5-slice volume where each slice has constant value:
        # slice 0 = 10, slice 1 = 20, slice 2 = 30, slice 3 = 40, slice 4 = 50
        vol = np.zeros((5, 32, 32), dtype=np.float32)
        lbl_vol = np.zeros((5, 32, 32), dtype=np.int16)
        for s in range(5):
            vol[s] = float(s + 1) * 10.0
            lbl_vol[s] = s  # label has value equal to slice index

        np.savez(tmp_dir / "volume_5.npz", image=vol, label=lbl_vol)
        df = pd.DataFrame([{"record_id": "volume_5", "subject_id": "patient_1", "image_path": "fake.nii.gz", "has_label": True}])

        # 1. 2D Dataset (in_channels=1)
        ds_2d = LgeLaxDataset(records=df, data_root=tmp_dir, cache_dir=tmp_dir, in_channels=1)
        assert len(ds_2d) == 5

        for s in range(5):
            item = ds_2d[s]
            img = item["image"].numpy()
            lbl = item["label"].numpy()
            assert img.shape == (1, 32, 32), f"Expected shape (1, 32, 32), got {img.shape}"
            assert np.allclose(img[0], float(s + 1) * 10.0)
            assert np.all(lbl == s)

        # 2. 2.5D Dataset (in_channels=3)
        ds_25d = LgeLaxDataset(records=df, data_root=tmp_dir, cache_dir=tmp_dir, in_channels=3)
        assert len(ds_25d) == 5

        # Slice 0 (Boundary: s=0): channels should be [s0, s0, s1] = [10, 10, 20]
        item0 = ds_25d[0]
        img0 = item0["image"].numpy()
        assert img0.shape == (3, 32, 32)
        assert np.allclose(img0[0], 10.0), f"Slice 0 prev_idx must clamp to s0 (10.0), got {img0[0,0,0]}"
        assert np.allclose(img0[1], 10.0), f"Slice 0 curr_idx must be s0 (10.0), got {img0[1,0,0]}"
        assert np.allclose(img0[2], 20.0), f"Slice 0 next_idx must be s1 (20.0), got {img0[2,0,0]}"
        # Backward diff must be 0 (edge clamping)
        assert np.allclose(img0[1] - img0[0], 0.0)
        # Forward diff must be 10.0
        assert np.allclose(img0[2] - img0[1], 10.0)

        # Slice 2 (Interior: s=2): channels should be [s1, s2, s3] = [20, 30, 40]
        item2 = ds_25d[2]
        img2 = item2["image"].numpy()
        assert img2.shape == (3, 32, 32)
        assert np.allclose(img2[0], 20.0)
        assert np.allclose(img2[1], 30.0)
        assert np.allclose(img2[2], 40.0)
        assert np.allclose(img2[1] - img2[0], 10.0)
        assert np.allclose(img2[2] - img2[1], 10.0)

        # Slice 4 (Boundary: s=4): channels should be [s3, s4, s4] = [40, 50, 50]
        item4 = ds_25d[4]
        img4 = item4["image"].numpy()
        assert img4.shape == (3, 32, 32)
        assert np.allclose(img4[0], 40.0), f"Slice 4 prev_idx must be s3 (40.0), got {img4[0,0,0]}"
        assert np.allclose(img4[1], 50.0), f"Slice 4 curr_idx must be s4 (50.0), got {img4[1,0,0]}"
        assert np.allclose(img4[2], 50.0), f"Slice 4 next_idx must clamp to s4 (50.0), got {img4[2,0,0]}"
        # Forward diff must be 0 (edge clamping)
        assert np.allclose(img4[2] - img4[1], 0.0)
        # Backward diff must be 10.0
        assert np.allclose(img4[1] - img4[0], 10.0)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_dataset_single_slice_volume_25d():
    """Adversarially verify 2.5D behavior on single-slice volume (D=1)."""
    logger.info("Running test_dataset_single_slice_volume_25d...")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        vol1 = np.full((1, 64, 64), 42.0, dtype=np.float32)
        lbl1 = np.full((1, 64, 64), 3, dtype=np.int16)
        np.savez(tmp_dir / "single_sl.npz", image=vol1, label=lbl1)

        df = pd.DataFrame([{"record_id": "single_sl", "subject_id": "p_single", "image_path": "fake.nii.gz", "has_label": True}])
        ds = LgeLaxDataset(records=df, data_root=tmp_dir, cache_dir=tmp_dir, in_channels=3)
        assert len(ds) == 1

        item = ds[0]
        img = item["image"].numpy()
        lbl = item["label"].numpy()
        assert img.shape == (3, 64, 64)
        # All 3 channels must be identical [s0, s0, s0]
        assert np.allclose(img[0], 42.0)
        assert np.allclose(img[1], 42.0)
        assert np.allclose(img[2], 42.0)
        assert np.all(lbl == 3)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_dataset_on_the_fly_nifti_boundary_clamping():
    """Verify boundary clamping on-the-fly directly from NIfTI files (uncached path)."""
    logger.info("Running test_dataset_on_the_fly_nifti_boundary_clamping...")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # Create synthetic NIfTI volume of shape (32, 32, 4) with spatial features
        nii_data = np.zeros((32, 32, 4), dtype=np.float32)
        nii_lbl = np.zeros((32, 32, 4), dtype=np.int16)
        x_grid, y_grid = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
        for z in range(4):
            nii_data[:, :, z] = float(z + 1) * 10.0 + (x_grid + y_grid) * 0.1
            nii_lbl[:, :, z] = z

        img_nii = nib.Nifti1Image(nii_data, np.eye(4))
        lbl_nii = nib.Nifti1Image(nii_lbl, np.eye(4))

        img_path = tmp_dir / "test_img.nii.gz"
        lbl_path = tmp_dir / "test_lbl.nii.gz"
        nib.save(img_nii, str(img_path))
        nib.save(lbl_nii, str(lbl_path))

        df = pd.DataFrame([{
            "record_id": "rec_nii",
            "subject_id": "subj_1",
            "image_path": "test_img.nii.gz",
            "label_path": "test_lbl.nii.gz",
            "has_label": True,
        }])

        # Test on-the-fly dataset with in_channels=3
        ds_nii = LgeLaxDataset(
            records=df,
            data_root=tmp_dir,
            cache_dir=None,  # Forces on-the-fly reading
            target_shape=(32, 32),
            in_channels=3,
        )
        assert len(ds_nii) == 4

        # Check boundary slice 0: channels must be [s0, s0, s1]
        item0 = ds_nii[0]
        img0 = item0["image"].numpy()
        assert img0.shape == (3, 32, 32)
        # Channel 0 (prev) and Channel 1 (curr) must be EXACTLY identical (clamped to s0)
        assert np.allclose(img0[0], img0[1])
        # Channel 2 (next) is s1, must differ from s0
        assert not np.allclose(img0[1], img0[2])

        # Check boundary slice 3: channels must be [s2, s3, s3]
        item3 = ds_nii[3]
        img3 = item3["image"].numpy()
        assert img3.shape == (3, 32, 32)
        # Channel 1 (curr) and Channel 2 (next) must be EXACTLY identical (clamped to s3)
        assert np.allclose(img3[1], img3[2])
        # Channel 0 (prev) is s2, must differ from s3
        assert not np.allclose(img3[0], img3[1])

        # Interior slice 2: channels are [s1, s2, s3] (all different)
        item2 = ds_nii[2]
        img2 = item2["image"].numpy()
        assert img2.shape == (3, 32, 32)
        assert not np.allclose(img2[0], img2[1])
        assert not np.allclose(img2[1], img2[2])

        # Also test on-the-fly in_channels=1
        ds_nii_1ch = LgeLaxDataset(
            records=df,
            data_root=tmp_dir,
            cache_dir=None,
            target_shape=(32, 32),
            in_channels=1,
        )
        assert len(ds_nii_1ch) == 4
        item2_1ch = ds_nii_1ch[2]
        img2_1ch = item2_1ch["image"].numpy()
        assert img2_1ch.shape == (1, 32, 32)
        # 1ch slice 2 must match channel 1 (curr) of 3ch slice 2
        assert np.allclose(img2_1ch[0], img2[1])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_full_pipeline_dataloader_integration():
    """Stress-test end-to-end integration: dataset -> sampler -> DataLoader -> multi-batch iteration."""
    logger.info("Running test_full_pipeline_dataloader_integration...")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        records = []
        for i in range(12):
            rec_id = f"case_{i}"
            # Multi-slice volume (4 slices)
            vol = np.random.randn(4, 64, 64).astype(np.float32)
            lbl = np.zeros((4, 64, 64), dtype=np.int16)
            if i % 3 == 0:
                lbl[1:3, 20:30, 20:30] = 3  # Scar
            elif i % 3 == 1:
                lbl[0:2, 10:40, 10:40] = 2  # Myocardium
            # i % 3 == 2 is pure background
            np.savez(tmp_dir / f"{rec_id}.npz", image=vol, label=lbl)
            records.append({
                "record_id": rec_id,
                "subject_id": f"pat_{i}",
                "image_path": f"{rec_id}.nii.gz",
                "has_label": True,
            })

        df = pd.DataFrame(records)
        ds = LgeLaxDataset(records=df, data_root=tmp_dir, cache_dir=tmp_dir, in_channels=3, augment=True)
        assert len(ds) == 12 * 4  # 48 slices

        sampler = build_rare_class_sampler(ds, rare_boost=2.5, foreground_boost=1.5)
        assert sampler is not None
        assert len(sampler.weights) == 48

        # Verify weights distribution:
        # scar slices should have weight 2.5, myo slices 1.5, background slices 1.0
        scar_slices_count = sum(1 for w in sampler.weights if w == 2.5)
        myo_slices_count = sum(1 for w in sampler.weights if w == 1.5)
        bg_slices_count = sum(1 for w in sampler.weights if w == 1.0)
        assert scar_slices_count > 0
        assert myo_slices_count > 0
        assert bg_slices_count > 0
        assert scar_slices_count + myo_slices_count + bg_slices_count == 48

        # Create DataLoader and iterate for 3 epochs
        loader = DataLoader(ds, batch_size=8, sampler=sampler, drop_last=False)
        total_samples_seen = 0
        for epoch in range(3):
            epoch_samples = 0
            for batch in loader:
                imgs = batch["image"]
                lbls = batch["label"]
                assert imgs.shape[1:] == (3, 64, 64)
                assert lbls.shape[1:] == (64, 64)
                assert not torch.isnan(imgs).any()
                epoch_samples += imgs.shape[0]
            assert epoch_samples == 48
            total_samples_seen += epoch_samples

        assert total_samples_seen == 48 * 3
        logger.info("Successfully completed 3 epochs with DataLoader + RareClassSampler!")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_sampler_custom_classes_and_boosts():
    """Test custom rare class specifications, secondary classes, and boost values."""
    logger.info("Running test_sampler_custom_classes_and_boosts...")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # 4 records:
        # rec_0: background (0)
        # rec_1: class 1 (LV)
        # rec_2: class 2 (Myo)
        # rec_3: class 4 (RV)
        for i, cls in enumerate([0, 1, 2, 4]):
            lbl = np.full((16, 16), cls, dtype=np.int16)
            np.savez(tmp_dir / f"rec_{i}.npz", image=np.zeros((16, 16)), label=lbl)

        df = pd.DataFrame([{"record_id": f"rec_{i}", "subject_id": f"s_{i}", "has_label": True} for i in range(4)])
        ds = DummyDataset(records=df, cache_dir=tmp_dir)

        # Specify class 4 as primary rare (boost 4.0), class 1 as secondary (boost 2.0)
        sampler = build_rare_class_sampler(
            ds,
            rare_classes=[4, 1],
            rare_boost=4.0,
            foreground_boost=2.0,
        )
        assert sampler is not None
        # rec_0 (bg): 1.0
        # rec_1 (cls 1): 2.0
        # rec_2 (cls 2 > 0): 2.0 (fallback foreground)
        # rec_3 (cls 4): 4.0 (primary rare)
        assert sampler.weights[0] == 1.0
        assert sampler.weights[1] == 2.0
        assert sampler.weights[2] == 2.0
        assert sampler.weights[3] == 4.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_sampler_sax_volume_dataset_integration():
    """Verify build_rare_class_sampler directly on 3D volume LgeSaxDataset."""
    logger.info("Running test_sampler_sax_volume_dataset_integration...")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # Create 3 volumes:
        # vol_0: pure background (all 0)
        # vol_1: contains class 2 (Myocardium)
        # vol_2: contains class 3 (Scar)
        v0 = np.zeros((192, 192, 16), dtype=np.float32)
        l0 = np.zeros((192, 192, 16), dtype=np.int16)
        np.savez(tmp_dir / "sax_0.npz", image=v0, label=l0)

        v1 = np.zeros((192, 192, 16), dtype=np.float32)
        l1 = np.zeros((192, 192, 16), dtype=np.int16)
        l1[80:100, 80:100, 5:10] = 2
        np.savez(tmp_dir / "sax_1.npz", image=v1, label=l1)

        v2 = np.zeros((192, 192, 16), dtype=np.float32)
        l2 = np.zeros((192, 192, 16), dtype=np.int16)
        l2[85:95, 85:95, 7:9] = 3
        np.savez(tmp_dir / "sax_2.npz", image=v2, label=l2)

        df = pd.DataFrame([
            {"record_id": "sax_0", "subject_id": "sub_0", "image_path": "fake0.nii.gz", "has_label": True},
            {"record_id": "sax_1", "subject_id": "sub_1", "image_path": "fake1.nii.gz", "has_label": True},
            {"record_id": "sax_2", "subject_id": "sub_2", "image_path": "fake2.nii.gz", "has_label": True},
        ])

        ds_sax = LgeSaxDataset(records=df, data_root=tmp_dir, cache_dir=tmp_dir)
        sampler = build_rare_class_sampler(ds_sax, rare_boost=3.0, foreground_boost=1.8)
        assert sampler is not None
        assert len(sampler.weights) == 3
        assert sampler.weights[0] == 1.0  # Background volume
        assert sampler.weights[1] == 1.8  # Myocardium volume
        assert sampler.weights[2] == 3.0  # Scar volume

        # Check DataLoader with 3D batch
        loader = DataLoader(ds_sax, batch_size=2, sampler=sampler)
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0]["image"].shape == (2, 1, 16, 192, 192)
        assert batches[1]["image"].shape == (1, 1, 16, 192, 192)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_25d_augmentation_chirality_and_multichannel_consistency():
    """Verify 2.5D multichannel augmentation preserves channel synchronization and zero chirality flipping."""
    logger.info("Running test_25d_augmentation_chirality_and_multichannel_consistency...")
    from training.dataset.lge_dataset import MedicalAugmentation2D

    aug = MedicalAugmentation2D(flip_prob=0.0, rotate_range_deg=10.0, gamma_range=(0.8, 1.2), intensity_scale=0.05)
    # Create 3-channel input where channel 0 and channel 1 are identical (boundary clamping slice 0)
    base_slice = np.random.rand(64, 64).astype(np.float32)
    next_slice = np.random.rand(64, 64).astype(np.float32)
    img_3ch = np.stack([base_slice, base_slice, next_slice], axis=0)  # (3, 64, 64)
    lbl = np.random.randint(0, 4, (64, 64), dtype=np.int16)

    # Apply augmentation multiple times
    for _ in range(10):
        img_out, lbl_out = aug(img_3ch, lbl)
        assert img_out.shape == (3, 64, 64)
        assert lbl_out.shape == (64, 64)
        # Channel 0 and Channel 1 must remain identically transformed
        assert np.allclose(img_out[0], img_out[1], atol=1e-5), "2.5D boundary channel synchronization broke during augmentation!"
        # Channel 2 should differ
        assert not np.allclose(img_out[1], img_out[2], atol=1e-3)


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING ADVERSARIAL STRESS SUITE: SAMPLER & DATASET (M3)")
    print("=" * 60)
    test_sampler_empty_and_single_dataset()
    print("  [PASS] test_sampler_empty_and_single_dataset")
    test_sampler_uniform_weights_scenarios()
    print("  [PASS] test_sampler_uniform_weights_scenarios")
    test_sampler_empirical_draw_distribution()
    print("  [PASS] test_sampler_empirical_draw_distribution")
    test_sampler_extreme_class_distribution()
    print("  [PASS] test_sampler_extreme_class_distribution")
    test_sampler_corrupted_and_missing_file_resilience()
    print("  [PASS] test_sampler_corrupted_and_missing_file_resilience")
    test_dataset_2d_vs_25d_boundary_clamping()
    print("  [PASS] test_dataset_2d_vs_25d_boundary_clamping")
    test_dataset_single_slice_volume_25d()
    print("  [PASS] test_dataset_single_slice_volume_25d")
    test_dataset_on_the_fly_nifti_boundary_clamping()
    print("  [PASS] test_dataset_on_the_fly_nifti_boundary_clamping")
    test_full_pipeline_dataloader_integration()
    print("  [PASS] test_full_pipeline_dataloader_integration")
    test_sampler_custom_classes_and_boosts()
    print("  [PASS] test_sampler_custom_classes_and_boosts")
    test_sampler_sax_volume_dataset_integration()
    print("  [PASS] test_sampler_sax_volume_dataset_integration")
    test_25d_augmentation_chirality_and_multichannel_consistency()
    print("  [PASS] test_25d_augmentation_chirality_and_multichannel_consistency")
    print("=" * 60)
    print("ALL ADVERSARIAL SAMPLER & DATASET TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)
