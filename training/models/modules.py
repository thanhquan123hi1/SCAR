"""Shared building blocks for ResUNet++.

Strictly implemented according to:
Debesh Jha et al., "ResUNet++: An Advanced Architecture for Medical Image Segmentation" (ISM 2019).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Squeeze_Excite_Block(nn.Module):
    """Channel-wise attention via global average pooling + FC gating with reduction ratio r=8."""

    def __init__(self, channel: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channel = max(channel // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Stem_Block(nn.Module):
    """Stem block (initial feature extraction): Conv3x3 -> BN -> ReLU -> Conv3x3 + 1x1 Shortcut + SE."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1, reduction: int = 8):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )
        self.se = Squeeze_Excite_Block(out_c, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv_branch(x) + self.shortcut(x))


class ResidualConv(nn.Module):
    """Pre-activation residual convolutional block with 1x1 shortcut and Squeeze-and-Excitation."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1, padding: int = 1, reduction: int = 8):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )
        if in_c != out_c or stride != 1:
            self.conv_skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.conv_skip = nn.Identity()

        self.se = Squeeze_Excite_Block(out_c, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv_block(x) + self.conv_skip(x))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (DeepLab v3+ style).
    
    Contains:
    - 1×1 convolution (replaces the incorrect rate=1 with kernel_size=3)
    - 3 atrous convolutions with rates [6, 12, 18]
    - Image-level global average pooling for global context
    """

    def __init__(self, in_dims: int, out_dims: int, rate: list[int] | None = None):
        super().__init__()
        if rate is None:
            rate = [1, 6, 12, 18]

        self.branches = nn.ModuleList()
        for r in rate:
            if r <= 1:
                # 1×1 convolution branch (standard ASPP)
                branch = nn.Sequential(
                    nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dims),
                    nn.ReLU(inplace=True),
                )
            else:
                # Atrous convolution branch
                branch = nn.Sequential(
                    nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=1, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_dims),
                    nn.ReLU(inplace=True),
                )
            self.branches.append(branch)

        # Image-level pooling branch for global context
        # Use GroupNorm(1) instead of BatchNorm because AdaptiveAvgPool2d(1) 
        # produces 1×1 spatial output, which causes BatchNorm to fail with batch_size=1.
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dims, out_dims, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_dims),
            nn.ReLU(inplace=True),
        )

        # +1 for image pooling branch
        self.output = nn.Conv2d((len(rate) + 1) * out_dims, out_dims, kernel_size=1)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [b(x) for b in self.branches]
        # Image-level features: pool → upsample to match spatial dims
        img_feat = self.image_pool(x)
        img_feat = nn.functional.interpolate(img_feat, size=x.shape[2:], mode="bilinear", align_corners=False)
        branch_outs.append(img_feat)
        out = torch.cat(branch_outs, dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample_(nn.Module):
    """Bilinear 2x upsampling."""

    def __init__(self, scale: int = 2):
        super().__init__()
        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale, align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class AttentionBlock(nn.Module):
    """Attention Gate (Oktay et al.) — filters skip connection using decoder gating signal.
    
    The gate signal from the decoder (high-level semantics) selectively amplifies
    relevant spatial regions in the encoder skip connection (fine-grained details).
    """

    def __init__(self, input_encoder: int, input_decoder: int, output_dim: int):
        super().__init__()
        # Project skip connection (encoder) to output_dim
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(input_encoder, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
        )
        # Project gating signal (decoder) to output_dim
        self.conv_decoder = nn.Sequential(
            nn.Conv2d(input_decoder, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
        )
        # Attention coefficient
        self.conv_attn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, 1, kernel_size=1),
            nn.Sigmoid(),  # Bound attention weights strictly to [0, 1]
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """x1: encoder skip connection; x2: decoder gating feature (upsampled to x1's size)."""
        # Upsample decoder to match skip connection resolution if needed
        if x2.shape[2:] != x1.shape[2:]:
            x2_up = nn.functional.interpolate(x2, size=x1.shape[2:], mode="bilinear", align_corners=False)
        else:
            x2_up = x2
        gate = self.conv_encoder(x1) + self.conv_decoder(x2_up)
        gate = self.conv_attn(gate)  # (B, 1, H, W) in range [0, 1]
        return gate * x1  # Filter the skip connection, NOT the decoder
