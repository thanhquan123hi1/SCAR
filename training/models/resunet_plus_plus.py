"""ResUNet++ Architecture for 2D/2.5D Medical Image Segmentation.

Strictly follows:
Debesh Jha et al., "ResUNet++: An Advanced Architecture for Medical Image Segmentation" (ISM 2019).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.models.modules import (
    ASPP,
    AttentionBlock,
    ResidualConv,
    Stem_Block,
    Upsample_,
)


class ResUNetPlusPlusEncoder(nn.Module):
    """ResUNet++ Shared Feature Extractor."""

    def __init__(self, in_channels: int = 3, filters: list[int] | None = None, reduction: int = 8):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128, 256, 512]

        self.filters = filters

        # Stem block (Input -> F0, no downsampling, has SE)
        self.stem = Stem_Block(in_channels, filters[0], stride=1, reduction=reduction)

        # 3 Encoder stages with Residual Convolutions + SE (downsampling via stride=2)
        self.res_conv1 = ResidualConv(filters[0], filters[1], stride=2, padding=1, reduction=reduction)
        self.res_conv2 = ResidualConv(filters[1], filters[2], stride=2, padding=1, reduction=reduction)
        self.res_conv3 = ResidualConv(filters[2], filters[3], stride=2, padding=1, reduction=reduction)

        # ASPP bridge at bottleneck with rates [1, 2, 4] for 24x24 / 32x32 feature maps
        self.aspp_bridge = ASPP(filters[3], filters[4], rate=[1, 2, 4])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.stem(x)          # (B, F0, H, W)
        x2 = self.res_conv1(x1)     # (B, F1, H/2, W/2)
        x3 = self.res_conv2(x2)     # (B, F2, H/4, W/4)
        x4 = self.res_conv3(x3)     # (B, F3, H/8, W/8)
        x5 = self.aspp_bridge(x4)   # (B, F4, H/8, W/8)
        return x1, x2, x3, x4, x5


class ResUNetPlusPlusDecoder(nn.Module):
    """Complete 4-Stage ResUNet++ Decoder with Attention Gates, ResidualConvs with SE, and ASPP output."""

    def __init__(self, filters: list[int] | None = None, num_classes: int = 5, reduction: int = 8):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128, 256, 512]

        # Stage 1: F4 (bridge) + F3 (skip4) -> F3 at H/8
        self.attn1 = AttentionBlock(filters[3], filters[4], filters[3])
        self.up_res_conv1 = ResidualConv(filters[4] + filters[3], filters[3], stride=1, padding=1, reduction=reduction)
        self.upsample1 = Upsample_(scale=2)

        # Stage 2: F3 (d1) + F2 (skip3) -> F2 at H/4
        self.attn2 = AttentionBlock(filters[2], filters[3], filters[2])
        self.up_res_conv2 = ResidualConv(filters[3] + filters[2], filters[2], stride=1, padding=1, reduction=reduction)
        self.upsample2 = Upsample_(scale=2)

        # Stage 3: F2 (d2) + F1 (skip2) -> F1 at H/2
        self.attn3 = AttentionBlock(filters[1], filters[2], filters[1])
        self.up_res_conv3 = ResidualConv(filters[2] + filters[1], filters[1], stride=1, padding=1, reduction=reduction)
        self.upsample3 = Upsample_(scale=2)

        # Stage 4: F1 (d3) + F0 (skip1) -> F0 at H
        self.attn4 = AttentionBlock(filters[0], filters[1], filters[0])
        self.up_res_conv4 = ResidualConv(filters[1] + filters[0], filters[0], stride=1, padding=1, reduction=reduction)

        # Multi-scale ASPP Output with 3 branches [1, 2, 4] + 1x1 Classification Head
        self.aspp_out = ASPP(filters[0], filters[0], rate=[1, 2, 4])
        self.output_layer = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final_conv = self.output_layer

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
    ) -> torch.Tensor:
        # Decoder Stage 1: bridge(x5) + attended skip(x4) at H/8
        attn_x4 = self.attn1(x4, x5)
        if x5.shape[2:] != attn_x4.shape[2:]:
            x5 = F.interpolate(x5, size=attn_x4.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([x5, attn_x4], dim=1)        # (B, F4+F3, H/8, W/8)
        d1 = self.up_res_conv1(d1)                  # (B, F3, H/8, W/8)
        d1 = self.upsample1(d1)                     # (B, F3, H/4, W/4)

        # Decoder Stage 2: d1 + attended skip(x3) at H/4
        attn_x3 = self.attn2(x3, d1)
        if d1.shape[2:] != attn_x3.shape[2:]:
            d1 = F.interpolate(d1, size=attn_x3.shape[2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d1, attn_x3], dim=1)        # (B, F3+F2, H/4, W/4)
        d2 = self.up_res_conv2(d2)                  # (B, F2, H/4, W/4)
        d2 = self.upsample2(d2)                     # (B, F2, H/2, W/2)

        # Decoder Stage 3: d2 + attended skip(x2) at H/2
        attn_x2 = self.attn3(x2, d2)
        if d2.shape[2:] != attn_x2.shape[2:]:
            d2 = F.interpolate(d2, size=attn_x2.shape[2:], mode="bilinear", align_corners=False)
        d3 = torch.cat([d2, attn_x2], dim=1)        # (B, F2+F1, H/2, W/2)
        d3 = self.up_res_conv3(d3)                  # (B, F1, H/2, W/2)
        d3 = self.upsample3(d3)                     # (B, F1, H, W)

        # Decoder Stage 4: d3 + attended skip(x1) at H
        attn_x1 = self.attn4(x1, d3)
        if d3.shape[2:] != attn_x1.shape[2:]:
            d3 = F.interpolate(d3, size=attn_x1.shape[2:], mode="bilinear", align_corners=False)
        d4 = torch.cat([d3, attn_x1], dim=1)        # (B, F1+F0, H, W)
        d4 = self.up_res_conv4(d4)                  # (B, F0, H, W)

        # Multi-scale ASPP Output & Head
        out = self.aspp_out(d4)
        return self.output_layer(out)


class ResUNetPlusPlus2D(nn.Module):
    """Standalone ResUNet++ model for single-view 2D / 2.5D segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        filters: list[int] | None = None,
        one_vs_rest: bool = False,
        reduction: int = 8,
        **kwargs,
    ):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128, 256, 512]

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.one_vs_rest = one_vs_rest
        out_channels = num_classes - 1 if one_vs_rest else num_classes

        self.encoder = ResUNetPlusPlusEncoder(in_channels=in_channels, filters=filters, reduction=reduction)
        self.decoder = ResUNetPlusPlusDecoder(filters=filters, num_classes=out_channels, reduction=reduction)

    def forward(self, x: torch.Tensor, dataset_type: str | None = None) -> torch.Tensor:
        x1, x2, x3, x4, x5 = self.encoder(x)
        return self.decoder(x1, x2, x3, x4, x5)


class MultiDatasetResUNetPlusPlus(nn.Module):
    """Multi-Dataset ResUNet++ with 1 Shared Encoder and view-specific decoders."""

    DATASET_CONFIGS_DEFAULT = {
        "2ch": 4,
        "4ch": 5,
        "sa": 5,
        "ras154": 2,
    }

    def __init__(
        self,
        in_channels: int = 3,
        filters: list[int] | None = None,
        num_classes_map: dict[str, int] | None = None,
        one_vs_rest: bool = False,
        reduction: int = 8,
        **kwargs,
    ):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128, 256, 512]

        self.num_classes_map = num_classes_map or self.DATASET_CONFIGS_DEFAULT
        self.one_vs_rest = one_vs_rest

        self.encoder = ResUNetPlusPlusEncoder(in_channels=in_channels, filters=filters, reduction=reduction)
        self.decoders = nn.ModuleDict({
            dtype: ResUNetPlusPlusDecoder(
                filters=filters,
                num_classes=num_cls - 1 if one_vs_rest else num_cls,
                reduction=reduction,
            )
            for dtype, num_cls in self.num_classes_map.items()
        })

    def forward(self, x: torch.Tensor, dataset_type: str = "sa") -> torch.Tensor:
        if dataset_type not in self.decoders:
            raise ValueError(f"Unknown dataset_type '{dataset_type}'. Available: {list(self.decoders.keys())}")
        x1, x2, x3, x4, x5 = self.encoder(x)
        return self.decoders[dataset_type](x1, x2, x3, x4, x5)
