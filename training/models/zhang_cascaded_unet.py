"""Zhang (2021) Cascaded 2D-3D Convolutional Neural Network for Cardiac MRI Segmentation.

Reference:
    - Zhang, Y., 2021. Cascaded convolutional neural network for automatic myocardial
      infarction segmentation from delayed-enhancement cardiac mri.
      Statistical Atlases and Computational Models of the Heart (STACOM/EMIDEC MICCAI).
    - Lalande, A. et al., 2022. Deep Learning methods for automatic evaluation of
      delayed enhancement-MRI. The results of the EMIDEC challenge. MedIA 2022.

Architecture Overview:
    1. Stage 1 (2D Preliminary Segmentation): Operates on intra-slice 2D features to
       eliminate inter-slice slice-gap & motion misalignment artifacts.
    2. Stage 2 (3D Volumetric Refinement): Takes concatenated (Original 3D MRI + 2D Coarse Probabilities)
       to leverage 3D volumetric continuity, refining small myocardial infarction (MI)
       and persistent microvascular obstruction (PMO) structures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Stage 1: 2D Intra-Slice Preliminary U-Net Components
# ==============================================================================

class ConvBlock2D(nn.Module):
    """(Conv2D -> InstanceNorm2D / BatchNorm2D -> LeakyReLU) * 2 with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_instance_norm: bool = True,
    ) -> None:
        super().__init__()
        norm_layer = nn.InstanceNorm2d if use_instance_norm else nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = norm_layer(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = norm_layer(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels, affine=True),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out = self.act2(out + res)
        return out


class DownBlock2D(nn.Module):
    """Downsampling block: MaxPool2d -> ConvBlock2D."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock2D(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock2D(nn.Module):
    """Upsampling block: ConvTranspose2d -> Concatenate -> ConvBlock2D."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2, bias=False
        )
        self.norm = nn.InstanceNorm2d(in_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv = ConvBlock2D(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.conv_trans(x)))
        if x.shape[2:] != skip.shape[2:]:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        out = torch.cat([x, skip], dim=1)
        return self.conv(out)


class UNet2DStage1(nn.Module):
    """2D Preliminary U-Net for Intra-Slice Coarse Segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        features: Optional[List[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.in_conv = ConvBlock2D(in_channels, features[0], dropout=dropout)
        self.down1 = DownBlock2D(features[0], features[1], dropout=dropout)
        self.down2 = DownBlock2D(features[1], features[2], dropout=dropout)
        self.down3 = DownBlock2D(features[2], features[3], dropout=dropout)

        self.bottleneck = ConvBlock2D(features[3], features[3] * 2, dropout=dropout)

        self.up3 = UpBlock2D(features[3] * 2, features[3], features[3], dropout=dropout)
        self.up2 = UpBlock2D(features[3], features[2], features[2], dropout=dropout)
        self.up1 = UpBlock2D(features[2], features[1], features[1], dropout=dropout)
        self.up0 = UpBlock2D(features[1], features[0], features[0], dropout=dropout)

        self.out_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.in_conv(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)

        b = self.bottleneck(s3)

        d3 = self.up3(b, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        return self.out_conv(d0)


# ==============================================================================
# Stage 2: 3D Inter-Slice Volumetric Refinement U-Net Components (nnU-Net Style)
# ==============================================================================

class ConvBlock3D(nn.Module):
    """(Conv3D -> InstanceNorm3D -> LeakyReLU) * 2 with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.residual = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out = self.act2(out + res)
        return out


class DownBlock3D(nn.Module):
    """MaxPool3d with anisotropic pool stride -> ConvBlock3D."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_stride: Tuple[int, int, int] = (2, 2, 2),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=pool_stride, stride=pool_stride)
        self.conv = ConvBlock3D(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock3D(nn.Module):
    """Upsample -> Concatenate -> ConvBlock3D."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        scale_factor: Tuple[int, int, int] = (2, 2, 2),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.conv_trans = nn.ConvTranspose3d(
            in_channels, in_channels, kernel_size=scale_factor, stride=scale_factor, bias=False
        )
        self.norm = nn.InstanceNorm3d(in_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv = ConvBlock3D(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.conv_trans(x)))
        if x.shape[2:] != skip.shape[2:]:
            diff_d = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            x = F.pad(
                x,
                [
                    diff_w // 2,
                    diff_w - diff_w // 2,
                    diff_h // 2,
                    diff_h - diff_h // 2,
                    diff_d // 2,
                    diff_d - diff_d // 2,
                ],
            )
        out = torch.cat([x, skip], dim=1)
        return self.conv(out)


class UNet3DStage2(nn.Module):
    """3D nnU-Net Style Volumetric Segmentation Network for Fine Refinement."""

    def __init__(
        self,
        in_channels: int = 6,  # 1 (MRI) + 5 (Coarse Probabilities)
        num_classes: int = 5,
        features: Optional[List[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.in_conv = ConvBlock3D(in_channels, features[0], dropout=dropout)

        self.down1 = DownBlock3D(features[0], features[1], pool_stride=(2, 2, 2), dropout=dropout)
        self.down2 = DownBlock3D(features[1], features[2], pool_stride=(2, 2, 2), dropout=dropout)
        self.down3 = DownBlock3D(features[2], features[3], pool_stride=(2, 2, 2), dropout=dropout)

        self.bottleneck = ConvBlock3D(features[3], features[3] * 2, dropout=dropout)

        self.up3 = UpBlock3D(features[3] * 2, features[3], features[3], scale_factor=(2, 2, 2), dropout=dropout)
        self.up2 = UpBlock3D(features[3], features[2], features[2], scale_factor=(2, 2, 2), dropout=dropout)
        self.up1 = UpBlock3D(features[2], features[1], features[1], scale_factor=(2, 2, 2), dropout=dropout)
        self.up0 = UpBlock3D(features[1], features[0], features[0], scale_factor=(2, 2, 2), dropout=dropout)

        self.out_conv = nn.Conv3d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.in_conv(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)

        b = self.bottleneck(s3)

        d3 = self.up3(b, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        return self.out_conv(d0)


# ==============================================================================
# Unified Zhang Cascaded 2D-3D Model Wrapper
# ==============================================================================

class ZhangCascadedUNet(nn.Module):
    """Full Cascaded 2D-3D Architecture (Zhang 2021).

    Executes:
        1. 2D Coarse Segmentation on slices (B, C, D, H, W) -> (B, num_classes, D, H, W)
        2. Softmax / Probability concatenation: (B, 1 + num_classes, D, H, W)
        3. 3D Fine Segmentation Refinement -> (B, num_classes, D, H, W)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        coarse_features: Optional[List[int]] = None,
        fine_features: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_one_hot: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_one_hot = use_one_hot

        # Stage 1: 2D U-Net
        self.stage1_2d = UNet2DStage1(
            in_channels=in_channels,
            num_classes=num_classes,
            features=coarse_features or [32, 64, 128, 256],
            dropout=dropout,
        )

        # Input channels for Stage 2: raw MRI (1) + Stage 1 output (num_classes)
        stage2_in_channels = in_channels + num_classes

        # Stage 2: 3D U-Net Refinement
        self.stage2_3d = UNet3DStage2(
            in_channels=stage2_in_channels,
            num_classes=num_classes,
            features=fine_features or [32, 64, 128, 256],
            dropout=dropout,
        )

    def _predict_coarse_volume(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 2D Stage 1 model slice-by-slice across 3D volume.

        Args:
            x: Tensor of shape (B, C, D, H, W)
        Returns:
            coarse_logits: (B, num_classes, D, H, W)
            coarse_probs: (B, num_classes, D, H, W)
        """
        b, c, d, h, w = x.shape
        # Permute & reshape: (B, C, D, H, W) -> (B * D, C, H, W)
        x_slices = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)

        coarse_logits_2d = self.stage1_2d(x_slices)  # (B * D, num_classes, H, W)

        # Reshape back: (B * D, num_classes, H, W) -> (B, D, num_classes, H, W) -> (B, num_classes, D, H, W)
        coarse_logits = coarse_logits_2d.reshape(b, d, self.num_classes, h, w).permute(0, 2, 1, 3, 4)

        if self.use_one_hot:
            pred_classes = torch.argmax(coarse_logits, dim=1)  # (B, D, H, W)
            coarse_probs = F.one_hot(pred_classes, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        else:
            coarse_probs = torch.softmax(coarse_logits, dim=1)

        return coarse_logits, coarse_probs

    def forward_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning both stage 1 (coarse) and stage 2 (fine) logits.

        Args:
            x: (B, 1, D, H, W)
        Returns:
            Dict containing 'coarse_logits', 'coarse_probs', and 'fine_logits'
        """
        coarse_logits, coarse_probs = self._predict_coarse_volume(x)

        # Concatenate raw 3D volume and 2D coarse predictions
        x_stage2 = torch.cat([x, coarse_probs], dim=1)  # (B, 1 + num_classes, D, H, W)

        fine_logits = self.stage2_3d(x_stage2)  # (B, num_classes, D, H, W)

        return {
            "coarse_logits": coarse_logits,
            "coarse_probs": coarse_probs,
            "fine_logits": fine_logits,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning fine segmentation logits (B, num_classes, D, H, W)."""
        outputs = self.forward_stages(x)
        return outputs["fine_logits"]


# Convenience alias for model registry
Cascaded2D3DUNet = ZhangCascadedUNet
