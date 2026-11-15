"""Decoder cascade and segmentation head modules."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                            padding=padding, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, 3, padding=1)
        self.conv2 = Conv2dReLU(out_channels, out_channels, 3, padding=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                raise ValueError(f"Decoder/skip mismatch: {x.shape} vs {skip.shape}")
            x = torch.cat((x, skip), dim=1)
        return self.conv2(self.conv1(x))


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_skip = config.n_skip
        skip_channels = [c if i < self.n_skip else 0
                         for i, c in enumerate(config.skip_channels)]
        in_channels = [config.fused_channels] + list(config.decoder_channels[:-1])
        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch, skip_ch)
            for in_ch, out_ch, skip_ch in zip(in_channels, config.decoder_channels,
                                             skip_channels)])

    def forward(self, fused, skips):
        for i, block in enumerate(self.blocks):
            fused = block(fused, skips[i] if i < self.n_skip else None)
        return fused


NO_DECODER_REFINEMENT = "none"
BOUNDARY_DECODER_REFINEMENT = "boundary"
DECODER_REFINEMENT_MODES = (NO_DECODER_REFINEMENT, BOUNDARY_DECODER_REFINEMENT)


def _gn_groups(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class BoundaryRefinementBlock(nn.Module):
    """Multi-scale 3x3 + dilated-3x3 residual refinement at full prediction resolution."""

    def __init__(self, channels: int):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        groups = _gn_groups(channels, 8)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202603 + channels)
            self.dw_local = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
            self.dw_dilated = nn.Conv2d(
                channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False
            )
            self.refine = nn.Sequential(
                nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=groups, num_channels=channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=groups, num_channels=channels),
            )
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = torch.cat((self.dw_local(x), self.dw_dilated(x)), dim=1)
        return F.relu(x + self.gamma * self.refine(ms), inplace=True)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size,
                                  padding=kernel_size // 2))
