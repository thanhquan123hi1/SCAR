"""3D Residual U-Net for LGE SAX Volumetric Segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_norm_layer_3d(norm_type: str, channels: int) -> nn.Module:
    """Return appropriate 3D normalization layer.
    
    GroupNorm (default, 8 groups) avoids biased statistics on anisotropic volumes (e.g. 10mm Z vs 1mm XY).
    InstanceNorm3d is retained as an option for isotropic volumes.
    """
    if norm_type == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    elif norm_type == "batch":
        return nn.BatchNorm3d(channels)
    elif norm_type == "group":
        num_groups = min(8, channels)
        while channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        raise ValueError(f"Unknown 3D norm_type: {norm_type}")


class ConvBlock3D(nn.Module):
    """(Conv3D -> Norm3D -> LeakyReLU) * 2 with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, norm_type: str = "group") -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _get_norm_layer_3d(norm_type, out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _get_norm_layer_3d(norm_type, out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.residual = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                _get_norm_layer_3d(norm_type, out_channels),
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
    """MaxPool3d or Strided Conv -> ConvBlock3D."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_stride: tuple[int, int, int] = (2, 2, 2),
        norm_type: str = "group",
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=pool_stride, stride=pool_stride)
        self.conv = ConvBlock3D(in_channels, out_channels, norm_type=norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock3D(nn.Module):
    """Upsample -> Concatenate -> ConvBlock3D."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        scale_factor: tuple[int, int, int] = (2, 2, 2),
        norm_type: str = "group",
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        # Reduce channels during upsample to avoid parameter bloat
        self.conv_trans = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor, bias=False
        )
        self.norm = _get_norm_layer_3d(norm_type, out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels, norm_type=norm_type)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.conv_trans(x)))
        # Pad if dimensions don't perfectly match
        if x.shape[2:] != skip.shape[2:]:
            diff_d = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2, diff_d // 2, diff_d - diff_d // 2])
        out = torch.cat([x, skip], dim=1)
        return self.conv(out)


class UNet3D(nn.Module):
    """3D Residual U-Net for LGE Cardiac SAX Volume Segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        features: list[int] | None = None,
        dropout: float = 0.1,
        norm_type: str = "group",
        **kwargs,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.in_conv = ConvBlock3D(in_channels, features[0], dropout=dropout, norm_type=norm_type)

        # Downsampling: anisotropic pooling for thick-slice SAX (192, 192, 16)
        self.down1 = DownBlock3D(features[0], features[1], pool_stride=(1, 2, 2), norm_type=norm_type)  # (16, 96, 96)
        self.down2 = DownBlock3D(features[1], features[2], pool_stride=(2, 2, 2), norm_type=norm_type)  # (8, 48, 48)
        self.down3 = DownBlock3D(features[2], features[3], pool_stride=(2, 2, 2), norm_type=norm_type)  # (4, 24, 24)
        self.down4 = DownBlock3D(features[3], features[3] * 2, pool_stride=(2, 2, 2), norm_type=norm_type)  # (2, 12, 12)

        # Upsampling
        self.up3 = UpBlock3D(features[3] * 2, features[3], features[3], scale_factor=(2, 2, 2), norm_type=norm_type)
        self.up2 = UpBlock3D(features[3], features[2], features[2], scale_factor=(2, 2, 2), norm_type=norm_type)
        self.up1 = UpBlock3D(features[2], features[1], features[1], scale_factor=(2, 2, 2), norm_type=norm_type)
        self.up0 = UpBlock3D(features[1], features[0], features[0], scale_factor=(1, 2, 2), norm_type=norm_type)

        # Final segmentation head
        self.out_conv = nn.Conv3d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        s0 = self.in_conv(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        b = self.down4(s3)

        d3 = self.up3(b, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        logits = self.out_conv(d0)
        return logits
