"""Automated Unit Tests for Zhang (2021) Cascaded 2D-3D Segmentation Method."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from training.loss.zhang_combo_loss import (
    ZhangCascadedLoss,
    ZhangComboLoss,
    multiclass_soft_dice_loss,
)
from training.models.zhang_cascaded_unet import (
    UNet2DStage1,
    UNet3DStage2,
    ZhangCascadedUNet,
)
from training.postprocessing.zhang_postprocess import (
    anatomical_clean_scar,
    remove_scattered_pixels,
    zhang_postprocess,
)


def test_zhang_model_architectures():
    print("Testing Zhang 2D Stage 1 forward pass...")
    stage1 = UNet2DStage1(in_channels=1, num_classes=5, features=[16, 32, 64, 128])
    x2d = torch.randn(4, 1, 64, 64)
    out2d = stage1(x2d)
    assert out2d.shape == (4, 5, 64, 64), f"Stage 1 shape mismatch: {out2d.shape}"
    print("  -> UNet2DStage1 PASSED!")

    print("Testing Zhang 3D Stage 2 forward pass...")
    stage2 = UNet3DStage2(in_channels=6, num_classes=5, features=[16, 32, 64, 128])
    x3d = torch.randn(2, 6, 8, 32, 32)
    out3d = stage2(x3d)
    assert out3d.shape == (2, 5, 8, 32, 32), f"Stage 2 shape mismatch: {out3d.shape}"
    print("  -> UNet3DStage2 PASSED!")

    print("Testing Full ZhangCascadedUNet forward pass...")
    model = ZhangCascadedUNet(
        in_channels=1,
        num_classes=5,
        coarse_features=[16, 32, 64, 128],
        fine_features=[16, 32, 64, 128],
    )
    x_vol = torch.randn(2, 1, 8, 32, 32)
    out_dict = model.forward_stages(x_vol)

    assert "coarse_logits" in out_dict
    assert "coarse_probs" in out_dict
    assert "fine_logits" in out_dict

    assert out_dict["coarse_logits"].shape == (2, 5, 8, 32, 32)
    assert out_dict["fine_logits"].shape == (2, 5, 8, 32, 32)

    # Standard forward
    fine_out = model(x_vol)
    assert fine_out.shape == (2, 5, 8, 32, 32)
    print("  -> ZhangCascadedUNet End-to-End Pipeline PASSED!")


def test_zhang_losses():
    print("Testing Zhang Combo Loss & Cascaded Loss...")
    combo_loss = ZhangComboLoss(
        num_classes=5,
        ce_weight=1.0,
        dice_weight=1.0,
        class_weights=[0.5, 1.0, 1.5, 3.0, 1.0],
    )
    logits = torch.randn(2, 5, 8, 32, 32, requires_grad=True)
    labels = torch.randint(0, 5, (2, 8, 32, 32)).long()

    l_val = combo_loss(logits, labels)
    assert torch.isfinite(l_val), "Loss is not finite!"
    l_val.backward()
    assert logits.grad is not None, "Gradients not computed!"
    print(f"  -> ZhangComboLoss PASSED! (Loss: {l_val.item():.4f})")

    cascaded_loss = ZhangCascadedLoss(num_classes=5, coarse_loss_weight=0.5)
    outputs = {
        "coarse_logits": torch.randn(2, 5, 8, 32, 32),
        "fine_logits": torch.randn(2, 5, 8, 32, 32),
    }
    c_loss_val = cascaded_loss(outputs, labels)
    assert torch.isfinite(c_loss_val)
    print(f"  -> ZhangCascadedLoss PASSED! (Loss: {c_loss_val.item():.4f})")


def test_zhang_postprocessing():
    print("Testing Zhang Postprocessing...")
    mask = np.zeros((16, 64, 64), dtype=np.int16)

    # Myocardium
    mask[4:12, 20:44, 20:44] = 2

    # Main contiguous scar inside myocardium
    mask[6:10, 25:35, 25:35] = 3

    # Scattered isolated noise pixels (size < 8) in background
    mask[0, 2:4, 2:4] = 3  # isolated 4 voxels
    mask[15, 60, 60] = 3   # 1 voxel

    cleaned = zhang_postprocess(mask, min_size_voxels=8, scar_class=3, myo_class=2)

    # Main scar should remain
    assert np.sum(cleaned[6:10, 25:35, 25:35] == 3) > 0, "Main scar was incorrectly removed"
    # Noise should be removed
    assert cleaned[0, 2, 2] == 0, "Scattered noise was not removed"
    assert cleaned[15, 60, 60] == 0, "Scattered noise was not removed"
    print("  -> Zhang Postprocessing Filter PASSED!")


if __name__ == "__main__":
    test_zhang_model_architectures()
    test_zhang_losses()
    test_zhang_postprocessing()
    print("\n=======================================================")
    print(">>> ALL ZHANG (2021) IMPLEMENTATION TESTS PASSED! <<<")
    print("=======================================================\n")
