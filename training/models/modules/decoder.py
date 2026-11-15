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
WIDE_RES_FPN_DECODER_REFINEMENT = "wide_res_fpn"
DECODER_REFINEMENT_MODES = (
    NO_DECODER_REFINEMENT,
    BOUNDARY_DECODER_REFINEMENT,
    WIDE_RES_FPN_DECODER_REFINEMENT,
)


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


class WideResidualFullConvDecoder(nn.Module):
    """Multi-stage wide full-3x3 residual skip enrichment + full-resolution 128x128 FPN head.

    Eliminates SlimDecoderBlock's 8x skip channel compression and lack of 3x3 spatial
    convolutions at 128x128 while preserving 100% step-0 equivalence with E5/E5-B1R via
    fork_rng and zero-initialized residual gates at all four decoder stages.
    """

    def __init__(
        self,
        skip_channels: tuple[int, int, int] = (512, 256, 64),
        dec_channels: tuple[int, int, int, int] = (128, 64, 32, 32),
    ):
        super().__init__()
        s0, s1, s2 = skip_channels
        d1, d2, d3, d4 = dec_channels
        w1, w2, w3, w4 = max(d1 * 2, 64), max(d2 * 2, 64), max(d3 * 2, 32), max(d4 * 2, 32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202633 + s0 + s1 + s2 + d1 + d2 + d3 + d4)
            self.stage1 = nn.Sequential(
                nn.Conv2d(d1 + s0, w1, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=_gn_groups(w1, 16), num_channels=w1),
                nn.ReLU(inplace=True),
                nn.Conv2d(w1, w1, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=_gn_groups(w1, 16), num_channels=w1),
                nn.ReLU(inplace=True),
                nn.Conv2d(w1, d1, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=_gn_groups(d1, 8), num_channels=d1),
            )
            self.gamma1 = nn.Parameter(torch.zeros(1, d1, 1, 1))

            self.stage2 = nn.Sequential(
                nn.Conv2d(d2 + s1, w2, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=_gn_groups(w2, 16), num_channels=w2),
                nn.ReLU(inplace=True),
                nn.Conv2d(w2, w2, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=_gn_groups(w2, 16), num_channels=w2),
                nn.ReLU(inplace=True),
                nn.Conv2d(w2, d2, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=_gn_groups(d2, 8), num_channels=d2),
            )
            self.gamma2 = nn.Parameter(torch.zeros(1, d2, 1, 1))

            self.stage3 = nn.Sequential(
                nn.Conv2d(d3 + s2, w3, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=_gn_groups(w3, 8), num_channels=w3),
                nn.ReLU(inplace=True),
                nn.Conv2d(w3, w3, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=_gn_groups(w3, 8), num_channels=w3),
                nn.ReLU(inplace=True),
                nn.Conv2d(w3, d3, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=_gn_groups(d3, 8), num_channels=d3),
            )
            self.gamma3 = nn.Parameter(torch.zeros(1, d3, 1, 1))

            fpn_in = d4 + d3 + d2
            self.fpn_proj = nn.Sequential(
                nn.Conv2d(fpn_in, w4, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=_gn_groups(w4, 8), num_channels=w4),
                nn.ReLU(inplace=True),
            )
            self.fpn_local = nn.Conv2d(w4, w4, kernel_size=3, padding=1, bias=False)
            self.fpn_dilated = nn.Conv2d(w4, w4, kernel_size=3, padding=2, dilation=2, groups=w4, bias=False)
            self.fpn_out = nn.Sequential(
                nn.GroupNorm(num_groups=_gn_groups(w4, 8), num_channels=w4),
                nn.ReLU(inplace=True),
                nn.Conv2d(w4, d4, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=_gn_groups(d4, 8), num_channels=d4),
            )
            self.gamma4 = nn.Parameter(torch.zeros(1, d4, 1, 1))

    def refine_stage1(self, d1: torch.Tensor, skip0: torch.Tensor) -> torch.Tensor:
        if skip0.shape[2:] != d1.shape[2:]:
            skip0 = F.interpolate(skip0, size=d1.shape[2:], mode="bilinear", align_corners=False)
        return F.relu(d1 + self.gamma1 * self.stage1(torch.cat((d1, skip0), dim=1)), inplace=True)

    def refine_stage2(self, d2: torch.Tensor, skip1: torch.Tensor) -> torch.Tensor:
        if skip1.shape[2:] != d2.shape[2:]:
            skip1 = F.interpolate(skip1, size=d2.shape[2:], mode="bilinear", align_corners=False)
        return F.relu(d2 + self.gamma2 * self.stage2(torch.cat((d2, skip1), dim=1)), inplace=True)

    def refine_stage3(self, d3: torch.Tensor, skip2: torch.Tensor) -> torch.Tensor:
        if skip2.shape[2:] != d3.shape[2:]:
            skip2 = F.interpolate(skip2, size=d3.shape[2:], mode="bilinear", align_corners=False)
        return F.relu(d3 + self.gamma3 * self.stage3(torch.cat((d3, skip2), dim=1)), inplace=True)

    def refine_stage4(self, d4: torch.Tensor, d3: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        u3 = F.interpolate(d3, size=d4.shape[2:], mode="bilinear", align_corners=False)
        u2 = F.interpolate(d2, size=d4.shape[2:], mode="bilinear", align_corners=False)
        h = self.fpn_proj(torch.cat((d4, u3, u2), dim=1))
        ms = self.fpn_local(h) + self.fpn_dilated(h)
        return F.relu(d4 + self.gamma4 * self.fpn_out(ms), inplace=True)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size,
                                  padding=kernel_size // 2))
