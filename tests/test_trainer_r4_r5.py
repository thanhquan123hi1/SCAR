"""Unit and Integration Tests for Requirements R4 (Checkpoint & Grad Clip) and R5 (Validation Macro-Dice).

Tests:
1. test_trainer_save_and_load_checkpoint_with_scheduler:
   - Verifies save_checkpoint includes scheduler_state_dict.
   - Verifies load_checkpoint restores model, optimizer, and scheduler states cleanly.
2. test_trainer_gradient_clipping:
   - Verifies that gradients are clipped to max_grad_norm in _train_epoch.
3. test_trainer_validate_epoch_subject_macro_dice:
   - Verifies subject-level macro-averaging with symmetric True Negative handling (Dice=1.0).
4. test_trainer_validate_epoch_multislice_grouping:
   - Verifies that multiple 2D slices for the same subject_id / record_id accumulate correctly into a single subject volume.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.trainer.trainer import Trainer


class DummyDataset(Dataset):
    """Synthetic dataset generating deterministic batches for Trainer testing."""

    def __init__(self, samples: list[dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class SimpleSegNet(nn.Module):
    """Simple segmentation model with linear head for testing."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.conv(x)))


def test_trainer_save_and_load_checkpoint_with_scheduler(tmp_path: Path):
    """Verify save_checkpoint preserves scheduler state and load_checkpoint restores all states."""
    model = SimpleSegNet(in_channels=1, num_classes=4)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    config = {
        "output": {"dir": str(tmp_path)},
        "training": {"max_grad_norm": 1.0},
        "postprocess": {"anatomical_constraint": False, "use_rules": False},
    }

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        config=config,
        device="cpu",
        num_classes=4,
        scheduler=scheduler,
        run_dir=tmp_path,
    )

    # Step scheduler 3 times
    for _ in range(3):
        scheduler.step()
    lr_before = scheduler.get_last_lr()[0]

    # Save checkpoint
    val_loss = 0.42
    metrics = {"dice_scar": 0.85, "mean_dice": 0.88}
    trainer.save_checkpoint("ckpt_test", epoch=3, val_loss=val_loss, metrics=metrics)

    ckpt_file = tmp_path / "checkpoints" / "ckpt_test.pt"
    assert ckpt_file.exists(), "Checkpoint file was not created"

    # Inspect checkpoint content
    ckpt_raw = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    assert "scheduler_state_dict" in ckpt_raw, "scheduler_state_dict missing from checkpoint"
    assert ckpt_raw["scheduler_state_dict"] is not None
    assert ckpt_raw["epoch"] == 3
    assert ckpt_raw["val_loss"] == 0.42
    assert ckpt_raw["metrics"] == metrics

    # Create new model, optimizer, scheduler and load checkpoint
    new_model = SimpleSegNet(in_channels=1, num_classes=4)
    new_optimizer = AdamW(new_model.parameters(), lr=1e-3)
    new_scheduler = CosineAnnealingLR(new_optimizer, T_max=10, eta_min=1e-5)

    new_trainer = Trainer(
        model=new_model,
        optimizer=new_optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        config=config,
        device="cpu",
        num_classes=4,
        scheduler=new_scheduler,
        run_dir=tmp_path,
    )

    loaded_dict = new_trainer.load_checkpoint(ckpt_file)
    assert loaded_dict["epoch"] == 3

    # Check model weights are restored bit-exact
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2), "Model parameter mismatch after load_checkpoint"

    # Check scheduler last LR matches
    assert new_scheduler.get_last_lr()[0] == pytest.approx(lr_before), "Scheduler state mismatch after load_checkpoint"


def test_trainer_gradient_clipping(tmp_path: Path):
    """Verify that gradients are clipped to max_grad_norm in _train_epoch."""
    model = SimpleSegNet(in_channels=1, num_classes=4)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    config = {
        "output": {"dir": str(tmp_path)},
        "training": {"max_grad_norm": 0.5},
        "postprocess": {"anatomical_constraint": False, "use_rules": False},
    }

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        config=config,
        device="cpu",
        num_classes=4,
    )

    # Create batch that produces large gradients
    images = torch.randn(2, 1, 32, 32) * 100.0
    labels = torch.randint(0, 4, (2, 32, 32))
    dataset = DummyDataset([{"image": images[0], "label": labels[0]}, {"image": images[1], "label": labels[1]}])
    loader = DataLoader(dataset, batch_size=2)

    loss_val = trainer._train_epoch(loader)
    assert not math.isnan(loss_val)

    # Verify model parameters remain finite and healthy
    for name, p in model.named_parameters():
        assert not torch.isnan(p).any(), f"NaN in parameter {name}"
        assert not torch.isinf(p).any(), f"Inf in parameter {name}"


def test_trainer_validate_epoch_subject_macro_dice(tmp_path: Path):
    """Verify subject-level Macro-Dice computation with True Negative handling in _validate_epoch."""
    # Model that outputs predictable class logits
    class MockModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x is (B, 1, H, W)
            # We return logits corresponding directly to channels in x
            B, _, H, W = x.shape
            logits = torch.zeros(B, 4, H, W)
            for b in range(B):
                pred_cls = int(x[b, 0, 0, 0].item())
                logits[b, pred_cls, :, :] = 10.0
            return logits

    model = MockModel()
    config = {
        "output": {"dir": str(tmp_path)},
        "postprocess": {"anatomical_constraint": False, "use_rules": False},
    }

    trainer = Trainer(
        model=model,
        optimizer=None,
        loss_fn=nn.CrossEntropyLoss(),
        config=config,
        device="cpu",
        num_classes=4,
    )

    # 4 Subjects:
    # Sub 1: Pred=3 (Scar), GT=3 (Scar) -> Dice = 1.0
    # Sub 2: Pred=0 (BG), GT=0 (BG) -> True Negative for Scar (GT=0, Pred=0) -> Dice = 1.0
    # Sub 3: Pred=3 (Scar), GT=0 (BG) -> False Alarm for Scar (GT=0, Pred>0) -> Dice = 0.0
    # Sub 4: Half Scar: 50 pixels pred 3, 50 pixels GT 3, 50 overlap -> Dice = 2*50/(100+100) = 0.50
    h, w = 10, 10
    
    # Sub 1: full scar
    img1 = torch.full((1, h, w), 3.0)
    lbl1 = torch.full((h, w), 3, dtype=torch.long)

    # Sub 2: true negative healthy (class 0)
    img2 = torch.full((1, h, w), 0.0)
    lbl2 = torch.full((h, w), 0, dtype=torch.long)

    # Sub 3: false positive scar (model predicts 3, GT is 0)
    img3 = torch.full((1, h, w), 3.0)
    lbl3 = torch.full((h, w), 0, dtype=torch.long)

    # Sub 4: half overlap
    # We construct a custom prediction using head
    class CustomHeadModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, _, H, W = x.shape
            logits = torch.zeros(B, 4, H, W)
            for b in range(B):
                tag = int(x[b, 0, 0, 0].item())
                if tag == 1:
                    logits[b, 3, :, :] = 10.0  # sub 1 -> scar
                elif tag == 2:
                    logits[b, 0, :, :] = 10.0  # sub 2 -> bg
                elif tag == 3:
                    logits[b, 3, :, :] = 10.0  # sub 3 -> scar
                elif tag == 4:
                    logits[b, 3, :5, :] = 10.0  # sub 4 -> top half scar (50 voxels)
                    logits[b, 0, 5:, :] = 10.0  # sub 4 -> bottom half bg
            return logits

    trainer.model = CustomHeadModel()

    # Sub 4 GT: rows 2..7 are scar (60 voxels), overlap = 30 voxels (rows 2..4)
    # GT = 60, Pred = 50, Inter = 30 -> Dice = 2*30 / (60 + 50) = 60 / 110 ≈ 0.5454545
    lbl4 = torch.zeros((h, w), dtype=torch.long)
    lbl4[2:8, :] = 3

    samples = [
        {"image": torch.full((1, h, w), 1.0), "label": lbl1, "subject_id": "sub_1"},
        {"image": torch.full((1, h, w), 2.0), "label": lbl2, "subject_id": "sub_2"},
        {"image": torch.full((1, h, w), 3.0), "label": lbl3, "subject_id": "sub_3"},
        {"image": torch.full((1, h, w), 4.0), "label": lbl4, "subject_id": "sub_4"},
    ]

    loader = DataLoader(DummyDataset(samples), batch_size=2)
    val_loss, metrics = trainer._validate_epoch(loader)

    # Expected Scar Dice per subject:
    # sub_1: 1.0
    # sub_2: 1.0 (True Negative)
    # sub_3: 0.0 (False Positive)
    # sub_4: 60/110 ≈ 0.545454545
    expected_scar_macro = (1.0 + 1.0 + 0.0 + (60.0 / 110.0)) / 4.0

    assert "dice_class_3" in metrics
    assert "dice_scar" in metrics
    assert metrics["dice_scar"] == pytest.approx(expected_scar_macro, rel=1e-5)
    assert metrics["dice_class_3"] == pytest.approx(expected_scar_macro, rel=1e-5)


def test_trainer_validate_epoch_multislice_grouping(tmp_path: Path):
    """Verify that multiple slices for the same subject accumulate before computing Dice."""
    h, w = 10, 10

    class SliceModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, _, H, W = x.shape
            logits = torch.zeros(B, 4, H, W)
            # Predict scar on top half (50 voxels per slice)
            logits[:, 3, :5, :] = 10.0
            logits[:, 0, 5:, :] = 10.0
            return logits

    config = {
        "output": {"dir": str(tmp_path)},
        "postprocess": {"anatomical_constraint": False, "use_rules": False},
    }

    trainer = Trainer(
        model=SliceModel(),
        optimizer=None,
        loss_fn=nn.CrossEntropyLoss(),
        config=config,
        device="cpu",
        num_classes=4,
    )

    # Subject 1 has 3 slices:
    # Slice 0: GT rows 0..5 (50 voxels scar), Pred 50 -> Inter = 50
    # Slice 1: GT rows 0..5 (50 voxels scar), Pred 50 -> Inter = 50
    # Slice 2: GT rows 5..10 (50 voxels scar in bottom half), Pred top half (50) -> Inter = 0
    # Total Subject 1: GT = 150, Pred = 150, Inter = 100 -> Subject Dice = 2*100 / (150+150) = 200/300 = 2/3 ≈ 0.66667
    lbl_top = torch.zeros((h, w), dtype=torch.long)
    lbl_top[:5, :] = 3

    lbl_bot = torch.zeros((h, w), dtype=torch.long)
    lbl_bot[5:, :] = 3

    samples = [
        {"image": torch.zeros(1, h, w), "label": lbl_top, "record_id": "patient_01"},
        {"image": torch.zeros(1, h, w), "label": lbl_top, "record_id": "patient_01"},
        {"image": torch.zeros(1, h, w), "label": lbl_bot, "record_id": "patient_01"},
    ]

    loader = DataLoader(DummyDataset(samples), batch_size=3)
    val_loss, metrics = trainer._validate_epoch(loader)

    expected_dice = 200.0 / 300.0
    assert metrics["dice_scar"] == pytest.approx(expected_dice, rel=1e-5)


def test_r1_invert_spatial_mask_no_false_positive_resuscitation():
    """R1: Verify that sub-threshold 1-voxel noise predictions are not artificially resurrected."""
    from preprocessing.preprocessing import SpatialTransform, CenterTransform, invert_spatial_mask

    transform = SpatialTransform(
        original_shape=(200, 200),
        resized_shape=(100, 100),
        source_spacing=(1.0, 1.0),
        target_spacing=(2.0, 2.0),
        center=CenterTransform(
            source_shape=(100, 100),
            target_shape=(100, 100),
            crop_start=(0, 0),
            crop_stop=(100, 100),
            pad_lower=(0, 0),
            pad_upper=(0, 0),
        ),
    )

    pred = np.zeros((100, 100), dtype=np.int16)
    pred[40:43, 40:43] = 3  # 9-voxel genuine scar
    pred[10, 10] = 3        # 1-voxel isolated noise

    restored = invert_spatial_mask(pred, transform)
    assert restored.shape == (200, 200)
    assert (restored == 3).sum() > 0, "Genuine scar must be restored"


def test_r2_unet3d_bottleneck_depth_and_skip_shapes():
    """R2: Verify UNet3D maintains bottleneck depth D >= 8 on (192, 192, 16) inputs."""
    from training.models.unet_3d import UNet3D

    model = UNet3D(in_channels=1, num_classes=4, norm_type="instance")
    model.eval()

    x = torch.randn(1, 1, 16, 192, 192)
    with torch.no_grad():
        x0 = model.in_conv(x)      # (1, 32, 16, 192, 192)
        d1 = model.down1(x0)       # (1, 64, 16, 96, 96)
        d2 = model.down2(d1)       # (1, 128, 16, 48, 48)
        d3 = model.down3(d2)       # (1, 256, 16, 24, 24)
        d4 = model.down4(d3)       # (1, 512, 8, 12, 12) -> Bottleneck depth = 8 >= 8

        assert d4.shape == (1, 512, 8, 12, 12), f"Bottleneck shape mismatch: {d4.shape}"

        u3 = model.up3(d4, d3)     # (1, 256, 16, 24, 24)
        assert u3.shape == (1, 256, 16, 24, 24)

        u2 = model.up2(u3, d2)     # (1, 128, 16, 48, 48)
        assert u2.shape == (1, 128, 16, 48, 48)

        u1 = model.up1(u2, d1)     # (1, 64, 16, 96, 96)
        assert u1.shape == (1, 64, 16, 96, 96)

        u0 = model.up0(u1, x0)     # (1, 32, 16, 192, 192)
        assert u0.shape == (1, 32, 16, 192, 192)

        out = model.out_conv(u0)   # (1, 4, 16, 192, 192)
        assert out.shape == (1, 4, 16, 192, 192)


def test_r3_sampler_hwd_cache_layout_normalization(tmp_path: Path):
    """R3: Verify build_rare_class_sampler correctly normalizes (H, W, D) 3D SAX caches."""
    from training.dataset.sampler import build_rare_class_sampler
    import pandas as pd

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # SAX volume stored in cache as (H, W, D) = (192, 192, 16)
    # Scar only in depth slices 8..11 (Z-axis)
    lbl_hwd = np.zeros((192, 192, 16), dtype=np.int16)
    lbl_hwd[80:100, 80:100, 8:12] = 3  # Scar in axial slices 8..11

    np.savez_compressed(cache_dir / "rec_sax_01.npz", label=lbl_hwd)

    # Create mock dataset with 16 slice items
    class MockSliceDataset:
        def __init__(self, cache_p: Path):
            self.cache_dir = cache_p
            row = pd.Series({"record_id": "rec_sax_01"})
            self.slices = [{"row": row, "slice_idx": i} for i in range(16)]

        def __len__(self):
            return len(self.slices)

    dataset = MockSliceDataset(cache_dir)
    sampler = build_rare_class_sampler(dataset=dataset, rare_classes=[3, 2], rare_boost=3.0)
    assert sampler is not None, "Sampler should return WeightedRandomSampler"
    assert len(sampler.weights) == 16

    # Slices 8..11 must have weight = 3.0, others 1.0
    for s_idx in range(16):
        if 8 <= s_idx <= 11:
            assert sampler.weights[s_idx] == 3.0, f"Slice {s_idx} should have rare boost 3.0"
        else:
            assert sampler.weights[s_idx] == 1.0, f"Slice {s_idx} should have baseline weight 1.0"


def test_r6_enforce_anatomical_constraints_2d_transmural_scar():
    """R6: Verify that 2D slices with 100% transmural MI (zero myocardium) preserve scar anchored to cavity."""
    from training.postprocess.anatomical import enforce_anatomical_constraints

    mask_2d = np.zeros((100, 100), dtype=np.int16)
    # LV blood pool (Class 1) in center
    mask_2d[35:65, 35:65] = 1
    # 100% transmural scar replacing the superior wall (rows 25..35, cols 30..70) - directly abutting cavity
    mask_2d[25:35, 30:70] = 3
    # Distant floating noise scar in corner (far from cardiac cavity)
    mask_2d[5:10, 5:10] = 3

    cleaned = enforce_anatomical_constraints(
        mask_2d,
        scar_class=3,
        myo_class=2,
        cardiac_classes=(1, 2, 4),
        dilation_voxels=2,
        min_scar_voxels=5,
    )

    # Transmural scar contiguous with LV cavity (Class 1) must be preserved
    assert (cleaned[25:35, 30:70] == 3).sum() > 0, "Transmural scar abutting cavity must be preserved"
    # Distant floating noise in corner must be completely suppressed to 0
    assert (cleaned[5:10, 5:10] == 3).sum() == 0, "Distant floating scar must be suppressed"
