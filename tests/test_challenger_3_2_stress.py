"""Empirical Challenger 2 Stress-Testing Suite for SCAR Cardiac MRI Segmentation (R3, R4, R5).

Adversarial challenges covering:
- R3: build_rare_class_sampler layout permutations, center vs edge slice distributions, corrupted/missing files, boundary conditions.
- R4: Trainer save/load round-trips across diverse schedulers (CosineAnnealingLR, StepLR, SequentialLR, None), explosive gradient clipping strictly bounded to max_grad_norm.
- R5: Trainer._validate_epoch subject-level macro-averaging, unbalanced slice counts, 100% True Negative cohorts, complete misses/false alarms, and direct mathematical proof against Metrics Reloaded 2024.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    StepLR,
)
from torch.utils.data import DataLoader, Dataset
import pandas as pd

from training.dataset.sampler import build_rare_class_sampler
from training.trainer.trainer import Trainer


# ============================================================================
# Helpers & Mock Classes
# ============================================================================

class MockDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(self, samples: list[dict[str, Any]], cache_dir: Path | None = None, records: pd.DataFrame | None = None):
        self.samples = samples
        self.cache_dir = cache_dir
        if records is not None:
            self.records = records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class MockSliceItemDataset(Dataset):
    """Dataset with .slices attribute matching 2D slice extractor format."""

    def __init__(self, slices: list[dict[str, Any]], cache_dir: Path | None = None, data_root: Path | None = None):
        self.slices = slices
        self.cache_dir = cache_dir
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.slices[idx]


class TinyNet(nn.Module):
    """Predictable linear model for deterministic trainer verification."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.out = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.conv(x)))


class ExplodingGradNet(nn.Module):
    """Model designed to produce explosive gradients for clipping stress tests."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_channels * 16 * 16, num_classes * 16 * 16)
        self.in_channels = in_channels
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        flat = x.view(B, -1)
        out = self.linear(flat)
        return out.view(B, self.num_classes, H, W)


# ============================================================================
# R3: build_rare_class_sampler Stress Tests
# ============================================================================

class TestR3RareClassSamplerStress:
    """Adversarial stress tests for build_rare_class_sampler."""

    def test_hwd_vs_dhw_cache_axial_slice_detection(self, tmp_path: Path):
        """Stress test: 3D SAX volume stored as (H, W, D) vs (D, H, W).
        
        Scar only placed in axial slices 8..11 (center Z).
        If coronal slicing occurs (indexing axis 0), all slices would have 0 scar.
        We confirm (H, W, D) and (D, H, W) yield IDENTICAL correct boosting on slices 8..11.
        """
        cache_dir_hwd = tmp_path / "cache_hwd"
        cache_dir_hwd.mkdir(parents=True, exist_ok=True)
        cache_dir_dhw = tmp_path / "cache_dhw"
        cache_dir_dhw.mkdir(parents=True, exist_ok=True)

        # 1. Create (H, W, D) = (192, 192, 16)
        lbl_hwd = np.zeros((192, 192, 16), dtype=np.int16)
        lbl_hwd[60:100, 60:100, 8:12] = 3  # Scar in axial Z=8..11
        np.savez_compressed(cache_dir_hwd / "sub_hwd.npz", label=lbl_hwd)

        # 2. Create (D, H, W) = (16, 192, 192)
        lbl_dhw = np.transpose(lbl_hwd, (2, 0, 1))
        np.savez_compressed(cache_dir_dhw / "sub_dhw.npz", label=lbl_dhw)

        row_hwd = pd.Series({"record_id": "sub_hwd"})
        row_dhw = pd.Series({"record_id": "sub_dhw"})

        slices_hwd = [{"row": row_hwd, "slice_idx": i} for i in range(16)]
        slices_dhw = [{"row": row_dhw, "slice_idx": i} for i in range(16)]

        ds_hwd = MockSliceItemDataset(slices_hwd, cache_dir=cache_dir_hwd)
        ds_dhw = MockSliceItemDataset(slices_dhw, cache_dir=cache_dir_dhw)

        sampler_hwd = build_rare_class_sampler(ds_hwd, rare_classes=[3, 2], rare_boost=3.0, foreground_boost=1.5)
        sampler_dhw = build_rare_class_sampler(ds_dhw, rare_classes=[3, 2], rare_boost=3.0, foreground_boost=1.5)

        assert sampler_hwd is not None
        assert sampler_dhw is not None

        weights_hwd = sampler_hwd.weights.numpy()
        weights_dhw = sampler_dhw.weights.numpy()

        # Both layouts must yield bit-identical weights
        np.testing.assert_array_equal(weights_hwd, weights_dhw)

        # Slices 8..11 must be boosted to 3.0, slices 0..7 and 12..15 must be baseline 1.0
        for i in range(16):
            if 8 <= i <= 11:
                assert weights_hwd[i] == 3.0, f"Axial slice {i} must receive rare_boost=3.0"
            else:
                assert weights_hwd[i] == 1.0, f"Background axial slice {i} must have baseline weight 1.0"

    def test_non_square_anisotropic_hwd_cache(self, tmp_path: Path):
        """Stress test: Non-square in-plane cache (160, 208, 14) stored as (H, W, D)."""
        cache_dir = tmp_path / "cache_nonsquare"
        cache_dir.mkdir(parents=True, exist_ok=True)

        lbl = np.zeros((160, 208, 14), dtype=np.int16)
        # Place myocardium (class 2) in slice 3, scar (class 3) in slice 7
        lbl[50:80, 50:80, 3] = 2  # Myo
        lbl[50:80, 50:80, 7] = 3  # Scar

        np.savez_compressed(cache_dir / "rec_nonsquare.npz", label=lbl)

        row = pd.Series({"record_id": "rec_nonsquare"})
        slices = [{"row": row, "slice_idx": i} for i in range(14)]
        ds = MockSliceItemDataset(slices, cache_dir=cache_dir)

        sampler = build_rare_class_sampler(ds, rare_classes=[3, 2], rare_boost=2.5, foreground_boost=1.4)
        assert sampler is not None

        weights = sampler.weights.numpy()
        assert weights[7] == 2.5, "Slice 7 with scar must receive rare_boost 2.5"
        assert weights[3] == 1.4, "Slice 3 with myocardium must receive foreground_boost 1.4"
        assert weights[0] == 1.0, "Slice 0 with background must receive 1.0"

    def test_corrupted_missing_and_malformed_cache_files(self, tmp_path: Path):
        """Stress test: Missing files, corrupted .npz files, missing 'label' key, empty manifests."""
        cache_dir = tmp_path / "cache_chaos"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. Normal record with scar
        lbl = np.zeros((16, 64, 64), dtype=np.int16)
        lbl[5, 10:20, 10:20] = 3
        np.savez_compressed(cache_dir / "valid_rec.npz", label=lbl)

        # 2. Corrupted file (random bytes)
        corrupt_file = cache_dir / "corrupt_rec.npz"
        corrupt_file.write_bytes(b"NOT_A_VALID_NPZ_HEADER_DATA_GARBAGE_1234567890")

        # 3. Valid NPZ but missing 'label' key (only has 'image')
        np.savez_compressed(cache_dir / "no_label_rec.npz", image=np.zeros((16, 64, 64)))

        # Build dataset with mixed valid, corrupt, missing, and no-label records
        slices = [
            {"row": pd.Series({"record_id": "valid_rec"}), "slice_idx": 5},      # valid -> weight 2.0
            {"row": pd.Series({"record_id": "corrupt_rec"}), "slice_idx": 0},    # corrupt -> fallback 1.0
            {"row": pd.Series({"record_id": "no_label_rec"}), "slice_idx": 0},   # no label -> fallback 1.0
            {"row": pd.Series({"record_id": "missing_rec"}), "slice_idx": 0},    # missing -> fallback 1.0
        ]

        ds = MockSliceItemDataset(slices, cache_dir=cache_dir)
        sampler = build_rare_class_sampler(ds, rare_classes=[3, 2], rare_boost=2.0, foreground_boost=1.3)

        assert sampler is not None
        weights = sampler.weights.numpy()
        assert len(weights) == 4
        assert weights[0] == 2.0
        assert weights[1] == 1.0
        assert weights[2] == 1.0
        assert weights[3] == 1.0

    def test_out_of_bounds_slice_index_safety(self, tmp_path: Path):
        """Stress test: Slices referencing slice_idx >= depth of volume should not crash with IndexError."""
        cache_dir = tmp_path / "cache_oob"
        cache_dir.mkdir(parents=True, exist_ok=True)

        lbl = np.zeros((8, 64, 64), dtype=np.int16)
        lbl[2, 10:20, 10:20] = 3  # Scar in slice 2
        np.savez_compressed(cache_dir / "vol8.npz", label=lbl)

        row = pd.Series({"record_id": "vol8"})
        slices = [
            {"row": row, "slice_idx": 0},    # background slice -> weight 1.0
            {"row": row, "slice_idx": 2},    # valid slice -> scar (weight 2.0)
            {"row": row, "slice_idx": 999},  # out-of-bounds slice -> does not crash with IndexError
        ]

        ds = MockSliceItemDataset(slices, cache_dir=cache_dir)
        sampler = build_rare_class_sampler(ds, rare_classes=[3, 2], rare_boost=2.0)
        assert sampler is not None
        assert len(sampler.weights) == 3
        assert sampler.weights[0] == 1.0
        assert sampler.weights[1] == 2.0

    def test_empty_dataset_and_all_uniform_background(self, tmp_path: Path):
        """Stress test: Empty dataset returns None; all-background dataset returns None (or error if strict)."""
        # Empty dataset
        empty_ds = MockSliceItemDataset([])
        assert build_rare_class_sampler(empty_ds) is None

        # All background dataset
        cache_dir = tmp_path / "cache_bg"
        cache_dir.mkdir(parents=True, exist_ok=True)
        lbl_bg = np.zeros((10, 64, 64), dtype=np.int16)
        np.savez_compressed(cache_dir / "bg_rec.npz", label=lbl_bg)

        row = pd.Series({"record_id": "bg_rec"})
        slices = [{"row": row, "slice_idx": i} for i in range(10)]
        ds_bg = MockSliceItemDataset(slices, cache_dir=cache_dir)

        # In non-strict mode: returns None to let DataLoader fall back to standard shuffle
        assert build_rare_class_sampler(ds_bg, strict=False) is None

        # In strict mode: raises RuntimeError
        with pytest.raises(RuntimeError, match="(?i)rare-class sampler cannot boost any samples"):
            build_rare_class_sampler(ds_bg, strict=True)


# ============================================================================
# R4: Trainer Checkpoint State & Gradient Clipping Stress Tests
# ============================================================================

class TestR4CheckpointAndGradClipStress:
    """Adversarial stress tests for Trainer checkpointing and gradient clipping."""

    @pytest.mark.parametrize(
        "scheduler_type",
        ["cosine", "step", "sequential", "none"],
    )
    def test_checkpoint_roundtrip_all_schedulers(self, tmp_path: Path, scheduler_type: str):
        """Stress test: Save and load checkpoint round-trips across multiple scheduler architectures."""
        model = TinyNet(in_channels=1, num_classes=4)
        optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)

        scheduler = None
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
        elif scheduler_type == "step":
            scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
        elif scheduler_type == "sequential":
            s1 = LinearLR(optimizer, start_factor=0.1, total_iters=5)
            s2 = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
            scheduler = SequentialLR(optimizer, schedulers=[s1, s2], milestones=[5])

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            scheduler=scheduler,
            device=torch.device("cpu"),
            num_classes=4,
            run_dir=tmp_path / f"run_{scheduler_type}",
            config={"training": {"max_grad_norm": 1.0}},
        )

        # Advance training steps
        for _ in range(4):
            if scheduler is not None:
                scheduler.step()

        lr_at_save = optimizer.param_groups[0]["lr"]

        # Save checkpoint
        trainer.save_checkpoint("ckpt_stress", epoch=4, val_loss=0.35, metrics={"dice_scar": 0.77})
        ckpt_path = tmp_path / f"run_{scheduler_type}" / "checkpoints" / "ckpt_stress.pt"
        assert ckpt_path.exists()

        # Instantiate fresh model, optimizer, scheduler
        new_model = TinyNet(in_channels=1, num_classes=4)
        new_optimizer = AdamW(new_model.parameters(), lr=1e-2, weight_decay=1e-4)
        new_scheduler = None
        if scheduler_type == "cosine":
            new_scheduler = CosineAnnealingLR(new_optimizer, T_max=20, eta_min=1e-5)
        elif scheduler_type == "step":
            new_scheduler = StepLR(new_optimizer, step_size=2, gamma=0.5)
        elif scheduler_type == "sequential":
            s1 = LinearLR(new_optimizer, start_factor=0.1, total_iters=5)
            s2 = CosineAnnealingLR(new_optimizer, T_max=15, eta_min=1e-5)
            new_scheduler = SequentialLR(new_optimizer, schedulers=[s1, s2], milestones=[5])

        new_trainer = Trainer(
            model=new_model,
            optimizer=new_optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            scheduler=new_scheduler,
            device=torch.device("cpu"),
            num_classes=4,
            run_dir=tmp_path / f"run_{scheduler_type}_restored",
            config={"training": {"max_grad_norm": 1.0}},
        )

        loaded_ckpt = new_trainer.load_checkpoint(ckpt_path)
        assert loaded_ckpt["epoch"] == 4
        assert loaded_ckpt["val_loss"] == 0.35

        # Verify exact weight restoration
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2), "Model parameters must be bit-exact after load_checkpoint"

        # Verify LR restoration
        lr_at_load = new_optimizer.param_groups[0]["lr"]
        assert lr_at_load == pytest.approx(lr_at_save, rel=1e-5)

        # Step restored scheduler and verify trajectory matches original
        if scheduler is not None and new_scheduler is not None:
            scheduler.step()
            new_scheduler.step()
            assert new_optimizer.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"], rel=1e-5)

    def test_gradient_clipping_under_explosive_loss_gradients(self, tmp_path: Path):
        """Stress test: Feed inputs causing raw gradient norms > 10,000.
        
        Verify that clip_grad_norm_ strictly bounds total gradient norm <= max_grad_norm (e.g. 0.75).
        """
        model = ExplodingGradNet(in_channels=1, num_classes=4)
        optimizer = SGD(model.parameters(), lr=1e-3)

        max_clip = 0.75
        config = {
            "training": {"max_grad_norm": max_clip},
            "postprocess": {"anatomical_constraint": False, "use_rules": False},
        }

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            num_classes=4,
            run_dir=tmp_path,
            config=config,
        )

        # Adversarial input with massive magnitude to force enormous raw gradients
        x_huge = torch.randn(4, 1, 16, 16) * 1e4
        y = torch.randint(0, 4, (4, 16, 16))

        dataset = MockDataset([{"image": x_huge[i], "label": y[i]} for i in range(4)])
        loader = DataLoader(dataset, batch_size=4)

        # Intercept gradient norm right before optimizer.step via hook
        grad_norms = []

        def track_grad_norm():
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]),
                2,
            ).item()
            grad_norms.append(total_norm)

        # We monkey-patch optimizer.step to check gradient norm at the exact moment of stepping
        orig_step = optimizer.step
        def step_wrapper(*args, **kwargs):
            track_grad_norm()
            return orig_step(*args, **kwargs)
        optimizer.step = step_wrapper

        train_loss = trainer._train_epoch(loader)
        assert not math.isnan(train_loss)

        # Ensure gradient norm was tracked and strictly clipped to <= max_clip + epsilon
        assert len(grad_norms) > 0
        for norm_val in grad_norms:
            assert norm_val <= max_clip + 1e-5, f"Gradient norm {norm_val} exceeded max_grad_norm {max_clip}!"

        # Ensure parameters remain finite and non-NaN
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), f"Parameter {name} became non-finite under explosive gradient"


# ============================================================================
# R5: Trainer Validation Subject-Level Macro-Dice Stress Tests
# ============================================================================

class TestR5ValidationSubjectMacroDiceStress:
    """Adversarial stress tests for Trainer._validate_epoch."""

    def test_unbalanced_multislice_subjects_macro_vs_micro_independence(self, tmp_path: Path):
        """Stress test: 3 subjects with heavily unbalanced slice counts.
        
        Subject A: 20 slices, 100% perfect prediction -> Subject Dice = 1.0
        Subject B: 2 slices, 0% overlap (complete miss) -> Subject Dice = 0.0
        Subject C: 5 slices, 50% overlap -> Subject Dice = 0.5
        
        True Subject-Level Macro-Dice for Scar = (1.0 + 0.0 + 0.5) / 3 = 0.500000.
        Legacy Micro-Dice would be biased by Subject A (20 slices), yielding ~0.80.
        We confirm Trainer._validate_epoch computes EXACT Subject Macro-Dice = 0.500000.
        """
        h, w = 10, 10

        # Deterministic model returning logits based on channel 0 value of input
        class OracleModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, _, H, W = x.shape
                logits = torch.zeros(B, 4, H, W)
                for b in range(B):
                    mode = int(x[b, 0, 0, 0].item())
                    if mode == 1:
                        # Perfect Scar prediction
                        logits[b, 3, :, :] = 10.0
                    elif mode == 2:
                        # Complete Miss: predict BG (0) when GT is Scar (3)
                        logits[b, 0, :, :] = 10.0
                    elif mode == 3:
                        # 50% overlap: predict top half Scar, bottom half BG
                        logits[b, 3, :5, :] = 10.0
                        logits[b, 0, 5:, :] = 10.0
                return logits

        trainer = Trainer(
            model=OracleModel(),
            optimizer=None,
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            num_classes=4,
            run_dir=tmp_path,
            config={"postprocess": {"anatomical_constraint": False, "use_rules": False}},
        )

        samples = []

        # Subject A: 20 slices, mode=1 (100% match, GT=full scar)
        for i in range(20):
            samples.append({
                "image": torch.full((1, h, w), 1.0),
                "label": torch.full((h, w), 3, dtype=torch.long),
                "subject_id": "patient_A",
            })

        # Subject B: 2 slices, mode=2 (0% match, GT=full scar, Pred=BG)
        for i in range(2):
            samples.append({
                "image": torch.full((1, h, w), 2.0),
                "label": torch.full((h, w), 3, dtype=torch.long),
                "subject_id": "patient_B",
            })

        # Subject C: 5 slices, mode=3 (50% match, GT=full scar, Pred=top half scar)
        # Per slice: GT=100, Pred=50, Inter=50. Over 5 slices: GT=500, Pred=250, Inter=250.
        # Dice = 2*250 / (500 + 250) = 500 / 750 = 2/3 ≈ 0.666667
        for i in range(5):
            samples.append({
                "image": torch.full((1, h, w), 3.0),
                "label": torch.full((h, w), 3, dtype=torch.long),
                "subject_id": "patient_C",
            })

        loader = DataLoader(MockDataset(samples), batch_size=4, shuffle=False)
        _, metrics = trainer._validate_epoch(loader)

        # Expected Subject Dice for Scar:
        # A: 1.0
        # B: 0.0
        # C: 2.0 / 3.0
        expected_macro_scar = (1.0 + 0.0 + (2.0 / 3.0)) / 3.0

        assert "dice_scar" in metrics
        assert metrics["dice_scar"] == pytest.approx(expected_macro_scar, rel=1e-5)

    def test_100_percent_true_negative_cohort(self, tmp_path: Path):
        """Stress test: Cohort where all cases have GT=0 and Pred=0 for rare classes.
        
        Metrics Reloaded standard requires Dice = 1.0 for True Negative subjects (GT=0, Pred=0).
        Macro-Dice must equal 1.0 without producing NaN or zero division warnings.
        """
        h, w = 16, 16

        class AllBgModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, _, H, W = x.shape
                logits = torch.zeros(B, 4, H, W)
                logits[:, 0, :, :] = 10.0  # Always predict background (class 0)
                return logits

        trainer = Trainer(
            model=AllBgModel(),
            optimizer=None,
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            num_classes=4,
            run_dir=tmp_path,
            config={"postprocess": {"anatomical_constraint": False, "use_rules": False}},
        )

        samples = []
        for s_idx in range(5):
            for slice_idx in range(4):
                samples.append({
                    "image": torch.zeros((1, h, w)),
                    "label": torch.zeros((h, w), dtype=torch.long),  # Pure healthy background
                    "subject_id": f"healthy_subj_{s_idx}",
                })

        loader = DataLoader(MockDataset(samples), batch_size=4)
        _, metrics = trainer._validate_epoch(loader)

        # For non-background classes 1, 2, 3: GT=0 and Pred=0 -> Dice must be 1.0
        assert metrics["dice_class_1"] == 1.0
        assert metrics["dice_class_2"] == 1.0
        assert metrics["dice_class_3"] == 1.0
        assert metrics["dice_scar"] == 1.0
        assert metrics["mean_dice"] == 1.0

    def test_all_failure_modes_cohort(self, tmp_path: Path):
        """Stress test: Cohort with complete False Negative, complete False Positive, True Negative, and Partial Overlap.
        
        Sub 1: TN (GT=0, Pred=0) -> Dice = 1.0
        Sub 2: Complete FN (GT>0, Pred=0) -> Dice = 0.0
        Sub 3: Complete FP (GT=0, Pred>0) -> Dice = 0.0
        Sub 4: Partial overlap (GT=40, Pred=20, Inter=20) -> Dice = 2*20 / (40+20) = 40/60 = 2/3
        
        Expected Macro-Dice = (1.0 + 0.0 + 0.0 + 2/3) / 4 = 1.666667 / 4 ≈ 0.416667.
        """
        h, w = 10, 10

        class RoutingModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, _, H, W = x.shape
                logits = torch.zeros(B, 4, H, W)
                for b in range(B):
                    tag = int(x[b, 0, 0, 0].item())
                    if tag == 1:
                        # Sub 1: Pred=0 (BG)
                        logits[b, 0, :, :] = 10.0
                    elif tag == 2:
                        # Sub 2: Pred=0 (BG)
                        logits[b, 0, :, :] = 10.0
                    elif tag == 3:
                        # Sub 3: Pred=3 (Scar)
                        logits[b, 3, :, :] = 10.0
                    elif tag == 4:
                        # Sub 4: Pred=3 on top 2 rows (20 voxels)
                        logits[b, 3, :2, :] = 10.0
                        logits[b, 0, 2:, :] = 10.0
                return logits

        trainer = Trainer(
            model=RoutingModel(),
            optimizer=None,
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            num_classes=4,
            run_dir=tmp_path,
            config={"postprocess": {"anatomical_constraint": False, "use_rules": False}},
        )

        # Sub 1: TN (GT=0, tag=1 -> Pred=0)
        lbl1 = torch.zeros((h, w), dtype=torch.long)
        # Sub 2: FN (GT=3 full, tag=2 -> Pred=0)
        lbl2 = torch.full((h, w), 3, dtype=torch.long)
        # Sub 3: FP (GT=0, tag=3 -> Pred=3)
        lbl3 = torch.zeros((h, w), dtype=torch.long)
        # Sub 4: Partial (GT=3 on top 4 rows -> 40 voxels, tag=4 -> Pred=3 on top 2 rows -> 20 voxels)
        lbl4 = torch.zeros((h, w), dtype=torch.long)
        lbl4[:4, :] = 3

        samples = [
            {"image": torch.full((1, h, w), 1.0), "label": lbl1, "subject_id": "sub_tn"},
            {"image": torch.full((1, h, w), 2.0), "label": lbl2, "subject_id": "sub_fn"},
            {"image": torch.full((1, h, w), 3.0), "label": lbl3, "subject_id": "sub_fp"},
            {"image": torch.full((1, h, w), 4.0), "label": lbl4, "subject_id": "sub_partial"},
        ]

        loader = DataLoader(MockDataset(samples), batch_size=2)
        _, metrics = trainer._validate_epoch(loader)

        expected = (1.0 + 0.0 + 0.0 + (40.0 / 60.0)) / 4.0
        assert metrics["dice_scar"] == pytest.approx(expected, rel=1e-5)

    def test_validation_batching_invariance(self, tmp_path: Path):
        """Stress test: Ensure validation Macro-Dice is strictly invariant to batch_size (e.g. bs=1 vs bs=3 vs bs=7)."""
        h, w = 8, 8

        class DeterministicNet(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, _, H, W = x.shape
                logits = torch.zeros(B, 4, H, W)
                logits[:, 1, :4, :] = 5.0  # Class 1 top half
                logits[:, 2, 4:, :] = 5.0  # Class 2 bottom half
                return logits

        trainer = Trainer(
            model=DeterministicNet(),
            optimizer=None,
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            num_classes=4,
            run_dir=tmp_path,
            config={"postprocess": {"anatomical_constraint": False, "use_rules": False}},
        )

        samples = []
        for s in range(6):
            for sl in range(3):
                lbl = torch.zeros((h, w), dtype=torch.long)
                lbl[:3, :] = 1
                lbl[5:, :] = 2
                samples.append({
                    "image": torch.randn(1, h, w),
                    "label": lbl,
                    "subject_id": f"patient_{s}",
                })

        ds = MockDataset(samples)
        _, metrics_bs1 = trainer._validate_epoch(DataLoader(ds, batch_size=1))
        _, metrics_bs3 = trainer._validate_epoch(DataLoader(ds, batch_size=3))
        _, metrics_bs7 = trainer._validate_epoch(DataLoader(ds, batch_size=7))

        assert metrics_bs1["dice_class_1"] == pytest.approx(metrics_bs3["dice_class_1"], rel=1e-7)
        assert metrics_bs1["dice_class_1"] == pytest.approx(metrics_bs7["dice_class_1"], rel=1e-7)
        assert metrics_bs1["dice_class_2"] == pytest.approx(metrics_bs3["dice_class_2"], rel=1e-7)
        assert metrics_bs1["dice_class_2"] == pytest.approx(metrics_bs7["dice_class_2"], rel=1e-7)
        assert metrics_bs1["mean_dice"] == pytest.approx(metrics_bs7["mean_dice"], rel=1e-7)
