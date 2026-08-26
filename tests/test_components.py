"""Automated verification test for all SCAR components."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from preprocessing.preprocessing import (
    invert_spatial_mask,
    preprocess_mask,
    preprocess_spatial,
)
from training.loss import build_loss
from training.metrics import (
    calculate_scar_metrics,
    dice_score,
    hd95_binary,
    iou_score,
)
from training.models import build_model


def test_preprocessing_invertibility():
    print("Testing Preprocessing Invertibility...")
    # 3D SAX
    raw_img = np.random.rand(180, 180, 12).astype(np.float32)
    raw_lbl = np.random.randint(0, 5, size=(180, 180, 12)).astype(np.int16)
    src_spacing = (1.2, 1.2, 8.0)
    tgt_spacing = (1.0, 1.0, 10.0)
    tgt_shape = (192, 192, 16)

    proc_img, transform = preprocess_spatial(
        raw_img,
        source_spacing=src_spacing,
        target_spacing=tgt_spacing,
        target_shape=tgt_shape,
        interpolation_order=1,
        intensity_percentiles=(0.95, 99.5),
    )
    assert proc_img.shape == tgt_shape, f"Expected {tgt_shape}, got {proc_img.shape}"

    proc_lbl = preprocess_mask(
        raw_lbl,
        source_spacing=src_spacing,
        target_spacing=tgt_spacing,
        target_shape=tgt_shape,
    )
    assert proc_lbl.shape == tgt_shape, f"Expected {tgt_shape}, got {proc_lbl.shape}"

    restored = invert_spatial_mask(proc_lbl, transform)
    assert restored.shape == raw_lbl.shape, f"Expected {raw_lbl.shape}, got {restored.shape}"
    print("  -> 3D SAX Preprocessing invertibility PASSED!")

    # 2D LAX Slice
    raw_slice_2d = np.random.rand(240, 240).astype(np.float32)
    src_spacing_2d = (1.3, 1.3)
    tgt_spacing_2d = (1.0, 1.0)
    tgt_shape_2d = (256, 256)
    proc_2d, trans_2d = preprocess_spatial(
        raw_slice_2d,
        source_spacing=src_spacing_2d,
        target_spacing=tgt_spacing_2d,
        target_shape=tgt_shape_2d,
        interpolation_order=1,
    )
    assert proc_2d.shape == (256, 256), f"Expected (256, 256), got {proc_2d.shape}"
    print("  -> 2D LAX Slice Preprocessing PASSED!")


def test_models_forward():
    print("Testing Model Forward Passes...")
    unet3d = build_model("unet_3d", in_channels=1, num_classes=5, features=[16, 32, 64, 128])
    x3d = torch.randn(2, 1, 16, 64, 64)
    out3d = unet3d(x3d)
    assert out3d.shape == (2, 5, 16, 64, 64), f"UNet3D shape mismatch: {out3d.shape}"
    print("  -> UNet3D Forward Pass PASSED!")

    unet2d = build_model("unet_2d", in_channels=1, num_classes=5, features=[16, 32, 64, 128])
    x2d = torch.randn(2, 1, 128, 128)
    out2d = unet2d(x2d)
    assert out2d.shape == (2, 5, 128, 128), f"UNet2D Forward Pass PASSED!"


def test_losses_and_metrics():
    print("Testing Losses and Clinical Metrics...")
    loss_fn = build_loss("ce_dice", num_classes=5)
    logits = torch.randn(2, 5, 16, 32, 32)
    labels = torch.randint(0, 5, (2, 16, 32, 32)).long()
    loss = loss_fn(logits, labels)
    assert torch.isfinite(loss), "Loss is not finite!"
    print("  -> CE+Dice Loss PASSED!")

    pred = np.zeros((32, 32, 10), dtype=np.int16)
    pred[10:20, 10:20, 2:8] = 2  # Myo
    pred[12:15, 12:15, 3:6] = 3  # Scar

    truth = np.zeros((32, 32, 10), dtype=np.int16)
    truth[10:20, 10:20, 2:8] = 2
    truth[12:16, 12:16, 3:6] = 3

    d = dice_score(pred, truth, num_classes=5)
    iou = iou_score(pred, truth, num_classes=5)
    hd = hd95_binary(pred == 3, truth == 3, spacing=(1.0, 1.0, 10.0))
    scar_meta = calculate_scar_metrics(pred, spacing=(1.0, 1.0, 10.0))

    assert d[3] > 0.5, f"Expected scar dice > 0.5, got {d[3]}"
    assert hd is not None and hd < 10.0, f"Expected HD95 < 10mm, got {hd}"
    assert scar_meta.scar_mass_g > 0, "Expected positive scar mass"
    print(f"  -> Scar Dice: {d[3]:.4f}, HD95: {hd:.2f} mm, Scar Mass: {scar_meta.scar_mass_g:.4f} g")
    print("  -> Metrics and Clinical Scar Quantification PASSED!")


if __name__ == "__main__":
    test_preprocessing_invertibility()
    test_models_forward()
    test_losses_and_metrics()
    print("\n>>> ALL UNIT TESTS PASSED SUCCESSFULLY! <<<")
