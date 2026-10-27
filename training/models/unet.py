"""Canonical U-Net 2D Architecture for Multi-Modal MRI Segmentation.

Extracted directly from Pytorch-UNet (milesial/Pytorch-UNet):
    Ronneberger, O., Fischer, P., & Brox, T. (2015).
    U-Net: Convolutional Networks for Biomedical Image Segmentation.
    MICCAI 2015, LNCS 9351, pp. 234-241.
"""
from __future__ import annotations

from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        self.bilinear = bilinear
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping feature channels to class logits"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def get_unet_config() -> ConfigDict:
    """Return default configuration for 2D U-Net."""
    config = ConfigDict()
    config.architecture = "unet"
    config.model_name = "U-Net"
    config.ablation = "UNET"
    config.num_classes = 4
    config.in_channels = 3
    config.base_channels = 64
    config.bilinear = False
    return config


class UNet2D(nn.Module):
    """Full assembly of Canonical U-Net for multi-modal myocardial pathology segmentation.

    Accepts 3 separate MRI modalities (LGE, C0, T2) each of shape (B, 1, H, W)
    or a combined 3-channel tensor (B, 3, H, W), and outputs 4-class logits (B, 4, H, W).
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        in_channels: int = 3,
        num_classes: int = 4,
        base_channels: int = 64,
        bilinear: bool = False,
        gradient_checkpointing: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        cfg = dict(config or {})
        self.n_channels = int(cfg.get("in_channels", in_channels))
        self.n_classes = int(cfg.get("num_classes", cfg.get("n_classes", num_classes)))
        self.base_channels = int(cfg.get("base_channels", base_channels))
        self.bilinear = bool(cfg.get("bilinear", bilinear))
        self.gradient_checkpointing = bool(
            cfg.get("gradient_checkpointing", gradient_checkpointing)
        )

        resolved = get_unet_config()
        resolved.update(cfg)
        resolved.in_channels = self.n_channels
        resolved.num_classes = self.n_classes
        resolved.base_channels = self.base_channels
        resolved.bilinear = self.bilinear
        resolved.gradient_checkpointing = self.gradient_checkpointing
        if "ablation" in kwargs:
            resolved.ablation = kwargs["ablation"]
        self.config = ConfigDict(resolved)

        bc = self.base_channels
        factor = 2 if self.bilinear else 1

        self.inc = DoubleConv(self.n_channels, bc)
        self.down1 = Down(bc, bc * 2)
        self.down2 = Down(bc * 2, bc * 4)
        self.down3 = Down(bc * 4, bc * 8)
        self.down4 = Down(bc * 8, (bc * 16) // factor)
        self.up1 = Up(bc * 16, (bc * 8) // factor, self.bilinear)
        self.up2 = Up(bc * 8, (bc * 4) // factor, self.bilinear)
        self.up3 = Up(bc * 4, (bc * 2) // factor, self.bilinear)
        self.up4 = Up(bc * 2, bc, self.bilinear)
        self.outc = OutConv(bc, self.n_classes)

        if self.gradient_checkpointing:
            self.use_checkpointing()

    def use_checkpointing(self) -> None:
        """Enable PyTorch activation checkpointing to save GPU memory."""
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    def _prepare_input(self, *inputs: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        if len(inputs) == 3:
            return torch.cat(inputs, dim=1)
        if len(inputs) == 1:
            x = inputs[0]
            if x.shape[1] == 1 and "c0" in kwargs and "t2" in kwargs:
                return torch.cat([x, kwargs["c0"], kwargs["t2"]], dim=1)
            return x
        if "lge" in kwargs and "c0" in kwargs and "t2" in kwargs:
            return torch.cat([kwargs["lge"], kwargs["c0"], kwargs["t2"]], dim=1)
        raise ValueError(
            f"UNet2D expected 3 modalities (separate or concatenated), received {len(inputs)} inputs"
        )

    def forward(self, *inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = self._prepare_input(*inputs, **kwargs)
        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"UNet2D expected {self.n_channels} input channels, received {x.shape[1]}"
            )

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def forward_for_loss(self, *inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(*inputs, **kwargs)

    def load_pretrained_encoders(self, path: str | None = None) -> None:
        """U-Net does not use pretrained hybrid encoders; weights train from scratch."""
        if path:
            print(f"Note: U-Net is trained from scratch. Pretrained weights ({path}) ignored.")
