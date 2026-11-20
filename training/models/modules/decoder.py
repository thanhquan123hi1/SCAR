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
HR_ANAT_CALIB_DECODER_REFINEMENT = "hr_anat_calib"
HR_RAW_EDGE_DECODER_REFINEMENT = "hr_raw_edge"
ANAT_RING_CONTEXT_DECODER_REFINEMENT = "anat_ring_context"
DECODER_REFINEMENT_MODES = (
    NO_DECODER_REFINEMENT,
    BOUNDARY_DECODER_REFINEMENT,
    WIDE_RES_FPN_DECODER_REFINEMENT,
    HR_ANAT_CALIB_DECODER_REFINEMENT,
    HR_RAW_EDGE_DECODER_REFINEMENT,
    ANAT_RING_CONTEXT_DECODER_REFINEMENT,
)

STATIC_HEAD = "static"
DYNAMIC_COND_HEAD = "dynamic_cond"
HEAD_MODES = (STATIC_HEAD, DYNAMIC_COND_HEAD)


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


class HRAnatCalibRefinement(nn.Module):
    """Full-resolution 128x128 modality-contrast stem with myocardium-masked intensity calibration.

    1. Predicts a soft myocardial wall mask p_ring from decoder stage d4.
    2. Computes per-slice mean/std of CINE, PSIR, and T2w weighted by p_ring (excluding background blood pool),
       producing myocardium-calibrated intensities that normalize cross-dataset scale differences.
    3. Extracts 128x128 3x3 features from raw + calibrated inter-modality contrast, masks them by p_ring,
       and refines d4 via a zero-initialized residual 3x3 fusion block before the segmentation head.
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        hr_mid = max(channels, 24)
        gate_mid = max(channels // 2, 16)
        gn_ch = _gn_groups(channels, 8)
        gn_hr = _gn_groups(hr_mid, 8)
        gn_gate = _gn_groups(gate_mid, 8)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202661 + channels)
            self.ring_gate = nn.Sequential(
                nn.Conv2d(channels, gate_mid, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=gn_gate, num_channels=gate_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_mid, 1, kernel_size=1, bias=True),
            )
            self.hr_stem = nn.Sequential(
                nn.Conv2d(9, hr_mid, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=gn_hr, num_channels=hr_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(hr_mid, hr_mid, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=gn_hr, num_channels=hr_mid),
                nn.ReLU(inplace=True),
            )
            self.fuse = nn.Sequential(
                nn.Conv2d(channels + hr_mid, channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=gn_ch, num_channels=channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=gn_ch, num_channels=channels),
            )
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def _masked_calibrate(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        w = mask.clamp(min=1e-4)
        w_sum = w.sum(dim=(2, 3), keepdim=True) + 1.0
        mean = (w * x).sum(dim=(2, 3), keepdim=True) / w_sum
        var = (w * (x - mean).square()).sum(dim=(2, 3), keepdim=True) / w_sum
        std = torch.sqrt(var.clamp(min=0.0) + eps).clamp(min=0.05)
        return ((x - mean) / std).clamp(-5.0, 5.0)

    @staticmethod
    def _first_channel_match(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        if x.shape[1] > 1:
            x = x[:, :1, :, :]
        if x.shape[2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return x.float()

    def forward(self, d4: torch.Tensor, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        hw = d4.shape[2:]
        c = self._first_channel_match(cine, hw)
        p = self._first_channel_match(psir, hw)
        t = self._first_channel_match(t2w, hw)

        p_ring = torch.sigmoid(self.ring_gate(d4.float()))
        cal_c = self._masked_calibrate(c, p_ring)
        cal_p = self._masked_calibrate(p, p_ring)
        cal_t = self._masked_calibrate(t, p_ring)

        hr_in = torch.cat(
            (c, p, t, p - c, p - t, t - p, cal_p - cal_c, cal_p - cal_t, cal_t - cal_p),
            dim=1,
        ).to(d4.dtype)
        hr_feat = self.hr_stem(hr_in) * p_ring.to(d4.dtype)
        delta = self.fuse(torch.cat((d4, hr_feat), dim=1))
        return F.relu(d4 + self.gamma * delta, inplace=True)


class RawModalHDRRefinement(nn.Module):
    """Full-resolution 128x128 raw 3-sequence stem + multi-scale residual decoder refinement.

    Supplies the missing 1mm spatial 3x3 + dilated context right before the head
    directly from aligned raw (CINE, PSIR, T2w) images without any hand-crafted
    modality subtractions or slice Z-scoring (avoiding Case3003/Case2030 distortion).
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        stem_ch = max(channels // 2, 16)
        gn_stem = _gn_groups(stem_ch, 8)
        gn_ch = _gn_groups(channels, 8)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202691 + channels)
            self.raw_stem = nn.Sequential(
                nn.Conv2d(3, stem_ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=gn_stem, num_channels=stem_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_ch, stem_ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=gn_stem, num_channels=stem_ch),
                nn.ReLU(inplace=True),
            )
            self.fuse_in = nn.Sequential(
                nn.Conv2d(channels + stem_ch, channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=gn_ch, num_channels=channels),
                nn.ReLU(inplace=True),
            )
            self.conv_local = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.conv_dilated = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False)
            self.ms_norm = nn.GroupNorm(num_groups=gn_ch, num_channels=channels)
            self.ms_relu = nn.ReLU(inplace=True)
            self.fuse_out = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=gn_ch, num_channels=channels),
            )
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, d4: torch.Tensor, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        hw = d4.shape[2:]
        c = HRAnatCalibRefinement._first_channel_match(cine, hw)
        p = HRAnatCalibRefinement._first_channel_match(psir, hw)
        t = HRAnatCalibRefinement._first_channel_match(t2w, hw)
        hr_feat = self.raw_stem(torch.cat((c, p, t), dim=1).to(d4.dtype))
        h = self.fuse_in(torch.cat((d4, hr_feat), dim=1))
        ms = self.ms_relu(self.ms_norm(self.conv_local(h) + self.conv_dilated(h)))
        delta = self.fuse_out(ms)
        return d4 + self.gamma * delta


class AnatRingContextRefinement(nn.Module):
    """Full-resolution 128x128 CINE-anchored myocardial ring & global peak-to-mean context refinement.

    Addresses two structural bottlenecks of SlimDecoderBlock's 1x1 final stage:
      1. Missing 1mm 3x3 boundary context for crisp myocardial ring borders (uses CINE only
         to avoid injecting bright LGE/T2w blood-pool glare outside the wall).
      2. Lack of slice-global context at 128x128: computes soft wall-gated mean and peak
         features across the entire myocardial ring so secondary intensity fluctuations on
         healthy opposite walls are suppressed relative to the slice's true lesion peak.
    Zero-initialized residual gate with fork_rng guarantees exact step-0 equivalence.
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        gn_ch = _gn_groups(channels, 8)
        gate_mid = max(channels // 2, 16)
        gn_gate = _gn_groups(gate_mid, 8)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202701 + channels)
            # 1. CINE-anchored full-resolution anatomical wall feature & soft ring mask
            self.anat_conv = nn.Sequential(
                nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=gn_ch, num_channels=channels),
                nn.ReLU(inplace=True),
            )
            self.ring_gate = nn.Sequential(
                nn.Conv2d(channels, gate_mid, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=gn_gate, num_channels=gate_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_mid, 1, kernel_size=1, bias=True),
            )
            # 2. Slice-global Peak-to-Mean modulation over the soft myocardial wall
            self.ctx_gate = nn.Sequential(
                nn.Conv2d(channels * 2, gate_mid, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_mid, channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            # 3. Multi-scale 1mm residual fusion
            self.fuse_in = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=gn_ch, num_channels=channels),
                nn.ReLU(inplace=True),
            )
            self.conv_local = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.conv_dilated = nn.Conv2d(
                channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False
            )
            self.ms_norm = nn.GroupNorm(num_groups=gn_ch, num_channels=channels)
            self.ms_relu = nn.ReLU(inplace=True)
            self.fuse_out = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=gn_ch, num_channels=channels),
            )
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, d4: torch.Tensor, cine: torch.Tensor) -> torch.Tensor:
        hw = d4.shape[2:]
        c = HRAnatCalibRefinement._first_channel_match(cine, hw).to(d4.dtype)
        f_anat = self.anat_conv(torch.cat((d4, c), dim=1))
        g_ring = torch.sigmoid(self.ring_gate(f_anat))

        # Wall-weighted global mean & masked peak in fp32 for bf16 stability
        d4_fp32 = d4.float()
        w_fp32 = g_ring.float()
        denom = w_fp32.sum(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        mu_wall = ((d4_fp32 * w_fp32).sum(dim=(2, 3), keepdim=True) / denom).to(d4.dtype)
        wall_feat = d4 * g_ring
        max_wall = wall_feat.amax(dim=(2, 3), keepdim=True)

        ctx = self.ctx_gate(torch.cat((mu_wall, max_wall - mu_wall), dim=1))
        rel_wall = (d4 - mu_wall) * g_ring * ctx

        h = self.fuse_in(torch.cat((f_anat, rel_wall), dim=1))
        ms = self.ms_relu(self.ms_norm(self.conv_local(h) + self.conv_dilated(h)))
        delta = self.fuse_out(ms)
        return d4 + self.gamma * delta


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size,
                                  padding=kernel_size // 2))


class DynamicSliceConditionedHead(nn.Module):
    """I-MMSeg-inspired slice-conditioned dynamic classification head + 128x128 spatial refinement.

    Distills I-MMSeg's ForwardPredictionHead into a lightweight, fully vectorized 2D module:
      1. Full-resolution 3x3 depthwise-separable residual refinement (fixes SlimDecoderBlock's
         lack of spatial convolutions at 128x128).
      2. Multi-scale GAP slice summary (fused + skip0/1/2) combined with learnable class
         prototypes generates per-slice, per-class 2-layer dynamic 1x1 classifier weights.
      3. Zero-initialized final dynamic layer & edge gate preserve exact step-0 equivalence
         with static head while allowing immediate non-zero gradients at step 0.
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_classes: int = 4,
        context_channels: tuple[int, ...] = (512, 512, 256, 64),
        ctx_dim: int = 64,
        precls_dim: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.ctx_dim = ctx_dim
        self.precls_dim = precls_dim

        gn_in = _gn_groups(in_channels, 8)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202651 + in_channels + num_classes)
            # 1. Lightweight 128x128 spatial edge refinement
            self.edge_refine = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=gn_in, num_channels=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=gn_in, num_channels=in_channels),
            )
            self.edge_gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

            # 2. Multi-scale slice GAP context encoders
            self.gap_projs = nn.ModuleList([
                nn.Sequential(
                    nn.GroupNorm(num_groups=_gn_groups(ch, 8), num_channels=ch),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(ch, ctx_dim, kernel_size=1, bias=True),
                )
                for ch in context_channels
            ])
            self.class_protos = nn.Parameter(torch.randn(num_classes, ctx_dim) * 0.05)

            # 3. Shared pre-classification features + slice-conditioned dynamic controller
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(num_groups=gn_in, num_channels=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, precls_dim, kernel_size=1, bias=True),
            )
            self.controller_shared = nn.Sequential(
                nn.Linear(ctx_dim * 2, ctx_dim),
                nn.ReLU(inplace=True),
            )
            # Layer 1 dynamic conv: (precls_dim -> precls_dim) + bias
            self.controller_l1 = nn.Linear(ctx_dim, precls_dim * precls_dim + precls_dim)
            # Layer 2 dynamic conv: (precls_dim -> 1) + bias, zero-init for exact step-0 equivalence
            self.controller_l2 = nn.Linear(ctx_dim, precls_dim + 1)
            nn.init.zeros_(self.controller_l2.weight)
            nn.init.zeros_(self.controller_l2.bias)

    def forward(
        self,
        d4: torch.Tensor,
        static_head: nn.Module,
        context_feats: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        d4_ref = F.relu(d4 + self.edge_gamma * self.edge_refine(d4), inplace=True)
        base_logits = static_head(d4_ref)

        # Multi-scale slice summary (B, ctx_dim)
        slice_ctx = sum(proj(feat).flatten(1) for proj, feat in zip(self.gap_projs, context_feats)) / len(self.gap_projs)
        b = d4.shape[0]
        k = self.num_classes
        p = self.precls_dim

        # Per-slice, per-class conditioning vector (B, K, 2*ctx_dim)
        s_exp = slice_ctx.unsqueeze(1).expand(b, k, self.ctx_dim)
        c_exp = self.class_protos.unsqueeze(0).expand(b, k, self.ctx_dim)
        cond = self.controller_shared(torch.cat((s_exp, c_exp), dim=-1))  # (B, K, ctx_dim)

        l1_params = self.controller_l1(cond).float()  # (B, K, p*p + p)
        l2_params = self.controller_l2(cond).float()  # (B, K, p + 1)
        w1 = l1_params[..., : p * p].reshape(b, k, p, p)
        b1 = l1_params[..., p * p :]  # (B, K, p)
        w2 = l2_params[..., :p]  # (B, K, p)
        b2 = l2_params[..., p]  # (B, K)

        precls = self.precls_conv(d4_ref).float()  # (B, p, H, W)
        # Vectorized dynamic 2-layer 1x1 conv across batch & classes
        h1 = F.relu(torch.einsum("bchw,bkdc->bkdhw", precls, w1) + b1[:, :, :, None, None], inplace=True)
        dyn_logits = torch.einsum("bkdhw,bkd->bkhw", h1, w2) + b2[:, :, None, None]
        return base_logits + dyn_logits.to(base_logits.dtype)


class ModalityAuxSupervisionHeads(nn.Module):
    """Training-only per-modality auxiliary heads & multi-scale contrastive projections.

    Never executed during eval(), so inference FLOPs overhead is strictly zero.
    Physics-aligned per-modality targets:
      - CINE (bSSFP): 2-class anatomical segmentation (bg vs myocardial_ring)
      - PSIR (LGE):   3-class segmentation (bg vs non_scar_myo vs scar)
      - T2w:          3-class segmentation (bg vs normal_myo vs AAR/edema_inclusive)
    Plus 3-scale (H/2, H/4, H/8) 64-dim embeddings + learnable class prototypes for
    I-MMSeg-style biology-aware supervised contrastive alignment.
    """

    def __init__(
        self,
        skip_channels: tuple[int, int, int] = (512, 256, 64),
        fpn_dim: int = 32,
        embed_dim: int = 64,
        num_classes: int = 4,
    ):
        super().__init__()
        self.skip_channels = tuple(skip_channels)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        s0, s1, s2 = skip_channels  # (512 @ H/8, 256 @ H/4, 64 @ H/2)
        gn_fpn = _gn_groups(fpn_dim, 8)
        gn_emb = _gn_groups(embed_dim, 8)

        def _make_fpn_head(out_classes: int) -> nn.ModuleDict:
            return nn.ModuleDict({
                "p0": nn.Conv2d(s0, fpn_dim, kernel_size=1, bias=False),
                "p1": nn.Conv2d(s1, fpn_dim, kernel_size=1, bias=False),
                "p2": nn.Conv2d(s2, fpn_dim, kernel_size=1, bias=False),
                "fuse": nn.Sequential(
                    nn.GroupNorm(num_groups=gn_fpn, num_channels=fpn_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, groups=fpn_dim, bias=False),
                    nn.Conv2d(fpn_dim, fpn_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=gn_fpn, num_channels=fpn_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(fpn_dim, out_classes, kernel_size=1, bias=True),
                ),
            })

        def _make_embed_proj(in_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, embed_dim, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=gn_emb, num_channels=embed_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
            )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202661 + s0 + s1 + s2)
            self.head_cine = _make_fpn_head(2)
            self.head_psir = _make_fpn_head(3)
            self.head_t2w = _make_fpn_head(3)
            # 3 scales ordered [high->low]: (skip2 @ H/2, skip1 @ H/4, skip0 @ H/8)
            scale_chs = (s2, s1, s0)
            self.embed_cine = nn.ModuleList([_make_embed_proj(c) for c in scale_chs])
            self.embed_psir = nn.ModuleList([_make_embed_proj(c) for c in scale_chs])
            self.embed_t2w = nn.ModuleList([_make_embed_proj(c) for c in scale_chs])
            self.class_protos = nn.Parameter(torch.randn(3, num_classes, embed_dim) * 0.1)

    @staticmethod
    def _run_fpn(head: nn.ModuleDict, skips: list[torch.Tensor], target_hw: tuple[int, int]) -> torch.Tensor:
        s0, s1, s2 = skips[0], skips[1], skips[2]
        hw_mid = s2.shape[2:]
        f = (
            F.interpolate(head["p0"](s0), size=hw_mid, mode="bilinear", align_corners=False)
            + F.interpolate(head["p1"](s1), size=hw_mid, mode="bilinear", align_corners=False)
            + head["p2"](s2)
        )
        logits_mid = head["fuse"](f)
        if logits_mid.shape[2:] != target_hw:
            logits_mid = F.interpolate(logits_mid, size=target_hw, mode="bilinear", align_corners=False)
        return logits_mid

    def forward(
        self,
        skips_cine: list[torch.Tensor],
        skips_psir: list[torch.Tensor],
        skips_t2w: list[torch.Tensor],
        target_hw: tuple[int, int],
    ) -> dict[str, object]:
        aux_cine = self._run_fpn(self.head_cine, skips_cine, target_hw)
        aux_psir = self._run_fpn(self.head_psir, skips_psir, target_hw)
        aux_t2w = self._run_fpn(self.head_t2w, skips_t2w, target_hw)
        # Embeddings ordered [H/2, H/4, H/8]
        emb_cine = [proj(s) for proj, s in zip(self.embed_cine, (skips_cine[2], skips_cine[1], skips_cine[0]))]
        emb_psir = [proj(s) for proj, s in zip(self.embed_psir, (skips_psir[2], skips_psir[1], skips_psir[0]))]
        emb_t2w = [proj(s) for proj, s in zip(self.embed_t2w, (skips_t2w[2], skips_t2w[1], skips_t2w[0]))]
        return {
            "aux_cine": aux_cine,
            "aux_psir": aux_psir,
            "aux_t2w": aux_t2w,
            "emb_cine": emb_cine,
            "emb_psir": emb_psir,
            "emb_t2w": emb_t2w,
            "class_protos": self.class_protos,
        }
