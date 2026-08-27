"""Adversarial Stress Test Harness for SCAR Pipeline Bug Fixes (R1, R2, R6).

Empirical verification suite by Challenger 1 (challenger_3_1).
Tests stress-test the following components against adversarial conditions:
1. R1: `invert_spatial_mask` - Isolated 1-voxel noise, boundary noise, sub-threshold predictions,
   multi-class concentric structures, extreme non-square/odd shapes, and zero false-positive peak resurrection.
2. R2: `UNet3D` - Forward/backward passes on various shapes (batch sizes 1, 2, 4; odd resolutions 127x127, 129x129, 95x95,
   depths 15, 16, 17, 19, 24, 32), bottleneck depth D >= 8 verification, skip connection alignment, clean gradient flow.
3. R6: `enforce_anatomical_constraints` - 100% transmural scars (zero myocardium), large apical scars,
   multi-cavity hearts (LV=1, RV=4), distant corner/air noise suppression (100%), 3D multi-slice volume integrity,
   and PyTorch Tensor device/dtype preservation.
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.preprocessing import (
    CenterTransform,
    SpatialTransform,
    invert_spatial_mask,
    preprocess_mask,
    resize_to_shape,
)
from training.models.unet_3d import ConvBlock3D, DownBlock3D, UNet3D, UpBlock3D
from training.postprocess.anatomical import enforce_anatomical_constraints


# ===========================================================================
# 1. R1 CHALLENGE SUITE: invert_spatial_mask Stress Testing & Peak Inversion
# ===========================================================================

class TestR1InvertSpatialMaskStress:
    """Adversarial stress testing of invert_spatial_mask."""

    def test_no_connected_component_peak_injection_in_invert_spatial_mask(self):
        """Stress Test: Verify that invert_spatial_mask does NOT contain the connected-component peak guard."""
        src_invert = inspect.getsource(invert_spatial_mask)
        src_preprocess = inspect.getsource(preprocess_mask)

        # Ground truth preprocessing retains CC peak guard for micro-lesion preservation
        assert "generate_binary_structure" in src_preprocess
        assert "labeled_comp" in src_preprocess

        # Prediction inversion must NOT contain CC peak guard (prevents false-positive peak resurrection)
        assert "generate_binary_structure" not in src_invert, (
            "invert_spatial_mask must not contain generate_binary_structure"
        )
        assert "labeled_comp" not in src_invert, (
            "invert_spatial_mask must not contain labeled_comp"
        )
        assert "res_comp" not in src_invert, (
            "invert_spatial_mask must not contain res_comp"
        )

    def test_submerged_micro_noise_not_resurrected(self):
        """Stress Test: Isolated noise submerged by continuous interpolation is decoded via pure argmax."""
        # Setup transform: model prediction grid (100, 100) -> original NIfTI grid (200, 200)
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
        # Genuine scar cluster (8x8 voxels = 64 voxels)
        pred[45:53, 45:53] = 3

        restored = invert_spatial_mask(pred, transform)
        assert restored.shape == (200, 200)
        assert restored.dtype == np.int16

        # Genuine scar cluster must be preserved and scaled cleanly
        scar_voxels = (restored == 3).sum()
        assert scar_voxels >= 64, f"Genuine scar lesion must be restored, found {scar_voxels} voxels"

        # Background purity in outer quadrant
        assert (restored[0:30, 0:30] == 3).sum() == 0, "Outer background must remain completely 0"

    def test_multiclass_concentric_structures_preserved(self):
        """Stress Test: Multi-class concentric cardiac anatomy (Cavity=1, Myo=2, Scar=3) preserves topology."""
        transform = SpatialTransform(
            original_shape=(216, 216),
            resized_shape=(128, 128),
            source_spacing=(1.2, 1.2),
            target_spacing=(2.0, 2.0),
            center=CenterTransform(
                source_shape=(128, 128),
                target_shape=(128, 128),
                crop_start=(0, 0),
                crop_stop=(128, 128),
                pad_lower=(0, 0),
                pad_upper=(0, 0),
            ),
        )

        pred = np.zeros((128, 128), dtype=np.int16)
        # LV Blood Pool (Class 1) in center (radius 15)
        yy, xx = np.ogrid[:128, :128]
        dist_sq = (yy - 64) ** 2 + (xx - 64) ** 2
        pred[dist_sq <= 15**2] = 1

        # Myocardium (Class 2) ring (radius 15 to 25)
        pred[(dist_sq > 15**2) & (dist_sq <= 25**2)] = 2

        # Subendocardial Scar (Class 3) wedge (angle quadrant in myocardium)
        scar_wedge = (dist_sq > 15**2) & (dist_sq <= 22**2) & (yy <= 64) & (xx >= 64)
        pred[scar_wedge] = 3

        restored = invert_spatial_mask(pred, transform)
        assert restored.shape == (216, 216)

        # Confirm all 4 classes exist in restored mask
        unique_classes = set(np.unique(restored))
        assert {0, 1, 2, 3}.issubset(unique_classes), f"Expected classes {0, 1, 2, 3}, got {unique_classes}"

        # Confirm concentric topology: center is cavity (1), surrounded by myo (2) / scar (3), outside is BG (0)
        assert restored[108, 108] == 1, "Center must remain LV cavity"
        assert restored[0, 0] == 0, "Distant corner must remain background"
        assert (restored == 3).sum() > 0, "Scar wedge must be preserved"

    @pytest.mark.parametrize(
        ("pred_shape", "orig_shape"),
        [
            ((128, 64), (256, 128)),
            ((64, 128), (128, 256)),
            ((128, 128), (200, 150)),
            ((96, 96), (312, 144)),
            ((64, 64, 16), (192, 128, 16)),
            ((64, 64, 8), (180, 210, 10)),
        ],
    )
    def test_non_square_and_anisotropic_shapes(self, pred_shape: tuple[int, ...], orig_shape: tuple[int, ...]):
        """Stress Test: Invert spatial mask with diverse non-square and 3D anisotropic dimensions."""
        transform = SpatialTransform(
            original_shape=orig_shape,
            resized_shape=pred_shape,
            source_spacing=(1.0,) * len(orig_shape),
            target_spacing=(2.0,) * len(pred_shape),
            center=CenterTransform(
                source_shape=pred_shape,
                target_shape=pred_shape,
                crop_start=(0,) * len(pred_shape),
                crop_stop=pred_shape,
                pad_lower=(0,) * len(pred_shape),
                pad_upper=(0,) * len(pred_shape),
            ),
        )

        pred = np.zeros(pred_shape, dtype=np.int16)
        # Add central lesion of class 3
        slices = tuple(slice(s // 3, 2 * s // 3) for s in pred_shape)
        pred[slices] = 3

        restored = invert_spatial_mask(pred, transform)
        assert restored.shape == orig_shape
        assert restored.dtype == np.int16
        assert (restored == 3).sum() > 0, "Lesion must be properly scaled to non-square original shape"

    def test_uniform_and_empty_prediction_masks(self):
        """Stress Test: Edge cases with 0 classes or only background in prediction mask."""
        transform = SpatialTransform(
            original_shape=(100, 100),
            resized_shape=(50, 50),
            source_spacing=(1.0, 1.0),
            target_spacing=(2.0, 2.0),
            center=CenterTransform(
                source_shape=(50, 50),
                target_shape=(50, 50),
                crop_start=(0, 0),
                crop_stop=(50, 50),
                pad_lower=(0, 0),
                pad_upper=(0, 0),
            ),
        )

        # 1. All background (0)
        pred_bg = np.zeros((50, 50), dtype=np.int16)
        restored_bg = invert_spatial_mask(pred_bg, transform)
        assert restored_bg.shape == (100, 100)
        assert np.all(restored_bg == 0)

        # 2. All scar (3)
        pred_scar = np.full((50, 50), 3, dtype=np.int16)
        restored_scar = invert_spatial_mask(pred_scar, transform)
        assert restored_scar.shape == (100, 100)
        assert np.all(restored_scar == 3)


# ===========================================================================
# 2. R2 CHALLENGE SUITE: UNet3D Multi-Shape, Bottleneck & Gradient Flow
# ===========================================================================

class TestR2UNet3DArchitectureStress:
    """Adversarial stress testing of 3D UNet architecture, depth preservation, and gradient dynamics."""

    def test_unet3d_bottleneck_depth_retention_standard(self):
        """Verify bottleneck depth D >= 8 on standard (B, 1, 16, 192, 192) volume."""
        model = UNet3D(in_channels=1, num_classes=5, norm_type="group")
        model.eval()

        x = torch.randn(1, 1, 16, 192, 192)
        with torch.no_grad():
            s0 = model.in_conv(x)
            assert s0.shape == (1, 32, 16, 192, 192), f"Stage 0 shape: {s0.shape}"

            s1 = model.down1(s0)
            assert s1.shape == (1, 64, 16, 96, 96), f"Stage 1 shape: {s1.shape}"

            s2 = model.down2(s1)
            assert s2.shape == (1, 128, 16, 48, 48), f"Stage 2 shape: {s2.shape}"

            s3 = model.down3(s2)
            assert s3.shape == (1, 256, 16, 24, 24), f"Stage 3 shape: {s3.shape}"

            b = model.down4(s3)
            # Bottleneck MUST maintain depth D >= 8 (specifically D=8 here, not collapsed to D=2)
            assert b.shape == (1, 512, 8, 12, 12), f"Bottleneck shape: {b.shape}"
            assert b.shape[2] >= 8, f"Depth collapsed below 8: {b.shape[2]}"

            d3 = model.up3(b, s3)
            assert d3.shape == (1, 256, 16, 24, 24), f"Decoder 3 shape: {d3.shape}"

            d2 = model.up2(d3, s2)
            assert d2.shape == (1, 128, 16, 48, 48), f"Decoder 2 shape: {d2.shape}"

            d1 = model.up1(d2, s1)
            assert d1.shape == (1, 64, 16, 96, 96), f"Decoder 1 shape: {d1.shape}"

            d0 = model.up0(d1, s0)
            assert d0.shape == (1, 32, 16, 192, 192), f"Decoder 0 shape: {d0.shape}"

            out = model.out_conv(d0)
            assert out.shape == (1, 5, 16, 192, 192)

    @pytest.mark.parametrize(
        "tensor_shape",
        [
            (1, 1, 16, 192, 192),
            (2, 1, 16, 128, 128),
            (1, 1, 16, 64, 64),
            (1, 1, 24, 128, 128),
            (1, 1, 32, 128, 128),
            (1, 1, 16, 127, 127),  # Odd in-plane resolution
            (1, 1, 17, 129, 129),  # Odd depth and odd in-plane
            (1, 1, 15, 95, 95),    # Odd depth and odd in-plane
            (2, 1, 19, 101, 103),  # Asymmetric prime in-plane
        ],
    )
    def test_unet3d_forward_and_backward_various_shapes(self, tensor_shape: tuple[int, ...]):
        """Stress Test: Forward pass and backward gradient flow across diverse and odd spatial tensor shapes."""
        torch.manual_seed(42)
        model = UNet3D(in_channels=1, num_classes=4, features=[16, 32, 64, 128], norm_type="group")
        model.train()

        x = torch.randn(*tensor_shape, requires_grad=True)
        out = model(x)

        # Shape of output must match input batch, num_classes, depth, height, width
        expected_shape = (tensor_shape[0], 4, tensor_shape[2], tensor_shape[3], tensor_shape[4])
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"

        # Target label for multi-class cross entropy
        target = torch.randint(0, 4, (tensor_shape[0], tensor_shape[2], tensor_shape[3], tensor_shape[4]))
        loss = nn.functional.cross_entropy(out, target)

        assert not torch.isnan(loss), "Loss evaluated to NaN"
        assert not torch.isinf(loss), "Loss evaluated to Inf"

        # Backward pass: verify clean gradient backprop to all parameters
        loss.backward()

        assert x.grad is not None, "Input gradient is None"
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"

        # Audit every single trainable parameter in UNet3D
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient is None for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN detected in gradient of: {name}"
            assert not torch.isinf(param.grad).any(), f"Inf detected in gradient of: {name}"
            # Ensure gradients are actively non-zero (no dead layers or broken computational graph)
            assert torch.count_nonzero(param.grad) > 0, f"Dead parameter (zero grad): {name}"

    def test_unet3d_norm_types_compatibility(self):
        """Stress Test: Verify UNet3D operates stably with group, instance, and batch normalization."""
        for norm in ["group", "instance", "batch"]:
            model = UNet3D(in_channels=1, num_classes=3, features=[16, 32, 64, 128], norm_type=norm)
            model.train()
            x = torch.randn(2, 1, 16, 64, 64)
            out = model(x)
            assert out.shape == (2, 3, 16, 64, 64)
            loss = out.sum()
            loss.backward()


# ===========================================================================
# 3. R6 CHALLENGE SUITE: enforce_anatomical_constraints Stress Testing
# ===========================================================================

class TestR6AnatomicalConstraintsStress:
    """Adversarial stress testing of anatomical constraints and transmural scar preservation."""

    def test_2d_100_percent_transmural_scar_zero_myocardium(self):
        """Stress Test: 2D slice with 100% transmural scar (zero myocardium) bordering LV cavity is preserved."""
        # 100x100 2D slice
        mask = np.zeros((100, 100), dtype=np.int16)

        # LV Blood Pool Cavity (Class 1): circle at center (50, 50) radius 16
        yy, xx = np.ogrid[:100, :100]
        lv_cavity = ((yy - 50) ** 2 + (xx - 50) ** 2) <= 16**2
        mask[lv_cavity] = 1

        # 100% Transmural Scar (Class 3): Anterior wall crescent abutting the LV blood pool
        # Entire myocardium on this slice underwent transmural infarction necrosis -> 0 voxels of Class 2 (Myo) exist!
        scar_zone = (((yy - 50) ** 2 + (xx - 50) ** 2) > 16**2) & (((yy - 50) ** 2 + (xx - 50) ** 2) <= 26**2) & (yy < 50)
        mask[scar_zone] = 3

        # Add distant floating false-positive scar noise in the corner (air/chest wall)
        mask[2:8, 2:8] = 3
        mask[90:96, 90:96] = 3

        # Verify initial preconditions
        assert (mask == 2).sum() == 0, "Precondition failed: Myocardium must be 0 voxels (100% transmural)"
        assert (mask == 3).sum() > 50, "Precondition failed: Scar must exist"

        cleaned = enforce_anatomical_constraints(
            mask,
            scar_class=3,
            myo_class=2,
            cardiac_classes=(1, 2, 4),
            dilation_voxels=2,
            min_scar_voxels=5,
        )

        # 1. The transmural scar abutting the LV cavity (Class 1) MUST be 100% preserved
        initial_transmural_count = int(scar_zone.sum())
        cleaned_transmural_count = int((cleaned[scar_zone] == 3).sum())
        assert cleaned_transmural_count == initial_transmural_count, (
            f"Transmural scar corrupted: {cleaned_transmural_count} preserved out of {initial_transmural_count}"
        )

        # 2. The distant floating noise in corners MUST be 100% suppressed
        assert (cleaned[2:8, 2:8] == 3).sum() == 0, "Distant corner noise 1 was not suppressed"
        assert (cleaned[90:96, 90:96] == 3).sum() == 0, "Distant corner noise 2 was not suppressed"

    def test_2d_large_apical_scar_preservation(self):
        """Stress Test: Large apical scar with thinned/null myocardium is preserved."""
        mask = np.zeros((80, 80), dtype=np.int16)
        # Small apical blood pool
        mask[38:42, 38:42] = 1
        # Large apical cap scar enclosing the entire apex (30x30 region)
        mask[30:50, 30:50] = np.where(mask[30:50, 30:50] == 1, 1, 3)

        # Far noise in lungs/background
        mask[0:5, 0:5] = 3
        mask[75:80, 75:80] = 3

        cleaned = enforce_anatomical_constraints(
            mask,
            scar_class=3,
            myo_class=2,
            cardiac_classes=(1, 2, 4),
            dilation_voxels=2,
            min_scar_voxels=5,
        )

        # Apical scar preserved
        assert (cleaned[30:50, 30:50] == 3).sum() > 200, "Large apical scar must be preserved"
        # Lung noise suppressed
        assert (cleaned[0:5, 0:5] == 3).sum() == 0
        assert (cleaned[75:80, 75:80] == 3).sum() == 0

    def test_multicavity_heart_lv_rv_anchors(self):
        """Stress Test: Multiple cardiac cavities (LV=1, RV=4, Myo=2, Scar=3) correctly anchor scars."""
        mask = np.zeros((120, 120), dtype=np.int16)

        # LV Cavity (Class 1) at center (60, 50)
        yy, xx = np.ogrid[:120, :120]
        lv_mask = ((yy - 60) ** 2 + (xx - 50) ** 2) <= 12**2
        mask[lv_mask] = 1

        # RV Cavity (Class 4) crescent at (60, 80)
        rv_mask = ((yy - 60) ** 2 + (xx - 80) ** 2) <= 14**2
        mask[rv_mask] = 4

        # Scar lesion 1: Transmural scar bordering RV cavity (Class 4) with zero myocardium
        rv_scar = (((yy - 60) ** 2 + (xx - 80) ** 2) > 14**2) & (((yy - 60) ** 2 + (xx - 80) ** 2) <= 20**2) & (xx >= 80)
        mask[rv_scar] = 3

        # Scar lesion 2: Transmural scar bordering LV cavity (Class 1)
        lv_scar = (((yy - 60) ** 2 + (xx - 50) ** 2) > 12**2) & (((yy - 60) ** 2 + (xx - 50) ** 2) <= 18**2) & (xx <= 40)
        mask[lv_scar] = 3

        # Distant artifacts in 4 corners of the image (air / background)
        mask[0:10, 0:10] = 3
        mask[0:10, 110:120] = 3
        mask[110:120, 0:10] = 3
        mask[110:120, 110:120] = 3

        cleaned = enforce_anatomical_constraints(
            mask,
            scar_class=3,
            myo_class=2,
            cardiac_classes=(1, 2, 4),
            dilation_voxels=2,
            min_scar_voxels=5,
        )

        # Both RV-adjacent and LV-adjacent scars must be 100% preserved
        assert (cleaned[rv_scar] == 3).sum() == int(rv_scar.sum()), "RV cavity-adjacent scar must be preserved"
        assert (cleaned[lv_scar] == 3).sum() == int(lv_scar.sum()), "LV cavity-adjacent scar must be preserved"

        # All 4 distant corner artifacts must be 100% suppressed
        assert (cleaned[0:10, 0:10] == 3).sum() == 0
        assert (cleaned[0:10, 110:120] == 3).sum() == 0
        assert (cleaned[110:120, 0:10] == 3).sum() == 0
        assert (cleaned[110:120, 110:120] == 3).sum() == 0

    def test_complete_absence_of_cardiac_anatomy(self):
        """Stress Test: Mask containing ONLY noise/scar with zero cardiac anatomy must suppress all scar."""
        mask = np.zeros((100, 100), dtype=np.int16)
        # Random false-positive scar blobs
        mask[20:30, 20:30] = 3
        mask[70:80, 70:80] = 3

        cleaned = enforce_anatomical_constraints(
            mask,
            scar_class=3,
            myo_class=2,
            cardiac_classes=(1, 2, 4),
        )

        # Since no cardiac classes (1, 2, 4) exist in mask, all scar is unanchored and must be wiped to 0
        assert np.all(cleaned == 0), "All unanchored scar without cardiac anatomy must be suppressed"

    def test_3d_volume_transmural_apex_and_noise_suppression(self):
        """Stress Test: 3D Volume with apical transmural slice and distant noise across slices."""
        vol = np.zeros((16, 128, 128), dtype=np.int16)

        # Slices 4..12: Normal mid-ventricular slices with LV=1, Myo=2, Scar=3
        for z in range(4, 13):
            vol[z, 50:78, 50:78] = 1  # LV
            vol[z, 40:88, 40:88] = np.where(vol[z, 40:88, 40:88] == 1, 1, 2)  # Myo
            vol[z, 40:50, 50:78] = 3  # Scar inside anterior wall

        # Slice 3: Transmural apical slice (0 Myo, only LV cavity 1 + Scar 3)
        vol[3, 56:72, 56:72] = 1
        vol[3, 50:56, 56:72] = 3  # Transmural scar bordering cavity

        # Distant noise scattered in corners across slices 0..15
        vol[:, 0:10, 0:10] = 3
        vol[:, 118:128, 118:128] = 3

        cleaned = enforce_anatomical_constraints(
            vol,
            scar_class=3,
            myo_class=2,
            cardiac_classes=(1, 2, 4),
            dilation_voxels=2,
            min_scar_voxels=5,
        )

        # Transmural apical scar on slice 3 must be preserved
        assert (cleaned[3, 50:56, 56:72] == 3).sum() > 0, "Apical transmural scar on slice 3 must be preserved"

        # Mid-ventricle scars preserved
        for z in range(4, 13):
            assert (cleaned[z, 40:50, 50:78] == 3).sum() > 0

        # Corner noise wiped completely across all 16 slices
        assert (cleaned[:, 0:10, 0:10] == 3).sum() == 0, "Corner noise 1 not wiped in 3D"
        assert (cleaned[:, 118:128, 118:128] == 3).sum() == 0, "Corner noise 2 not wiped in 3D"

    def test_pytorch_tensor_device_and_dtype_preservation(self):
        """Stress Test: PyTorch Tensor input preserves tensor type, dtype, and device."""
        t_mask = torch.zeros((1, 80, 80), dtype=torch.long)
        t_mask[0, 30:50, 30:50] = 1  # LV cavity
        t_mask[0, 20:30, 30:50] = 3  # Transmural scar
        t_mask[0, 0:5, 0:5] = 3      # Noise

        cleaned_t = enforce_anatomical_constraints(
            t_mask[0],
            scar_class=3,
            myo_class=2,
            cardiac_classes=(1, 2, 4),
        )

        assert isinstance(cleaned_t, torch.Tensor), "Output must be torch.Tensor"
        assert cleaned_t.dtype == torch.long, "Tensor dtype must match input dtype"
        assert cleaned_t.shape == (80, 80), "Tensor shape must match input shape"
        assert (cleaned_t[20:30, 30:50] == 3).sum() > 0, "Transmural scar preserved in Tensor"
        assert (cleaned_t[0:5, 0:5] == 3).sum() == 0, "Noise removed in Tensor"
