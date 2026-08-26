"""2D Residual U-Net for LGE Multi-view 2D Slice Segmentation (2CH, 4CH, RAS, SAX slices)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    """(Conv2D -> BatchNorm2D -> LeakyReLU) * 2 with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
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
    """MaxPool2d -> ConvBlock2D."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock2D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock2D(nn.Module):
    """Upsample -> Concatenate -> ConvBlock2D."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        # Reduce channels during upsample to avoid parameter bloat
        self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv = ConvBlock2D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.conv_trans(x)))
        if x.shape[2:] != skip.shape[2:]:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        out = torch.cat([x, skip], dim=1)
        return self.conv(out)


class UNet2D(nn.Module):
    """2D Residual U-Net for LGE Multi-view 2D Segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        features: list[int] | None = None,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.in_conv = ConvBlock2D(in_channels, features[0], dropout=dropout)

        self.down1 = DownBlock2D(features[0], features[1])
        self.down2 = DownBlock2D(features[1], features[2])
        self.down3 = DownBlock2D(features[2], features[3])
        self.down4 = DownBlock2D(features[3], features[3] * 2)

        self.up3 = UpBlock2D(features[3] * 2, features[3], features[3])
        self.up2 = UpBlock2D(features[3], features[2], features[2])
        self.up1 = UpBlock2D(features[2], features[1], features[1])
        self.up0 = UpBlock2D(features[1], features[0], features[0])

        self.out_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
