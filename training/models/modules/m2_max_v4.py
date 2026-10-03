"""M2-Max-V4: Dual-Domain Pathology Routing with Multi-Scale Spatial-Anatomy Gating.

Major Innovations over M2-Max-V2:
1. Dual-Domain (Spatial + Channel) Pathology Disambiguation:
   Extends cooperative pathology interaction with both channel-selective and spatial-selective
   gates (c_gate * s_gate). Channel gates route core scar features vs salvageable peri-infarct
   features to their respective representations, while spatial gates refine lesion boundaries.
2. Spatial-Anatomy Multi-Scale Skip Gating (SpatialSkipGateFusion):
   Replaces 1x1 skip gating with 3x3 depthwise-separable anatomical gating on CINE, providing
   myocardial border context at high-resolution decoder stages and suppressing extracardiac noise.
3. Unimodal Modality Anchors (PSIR and T2w identity skip preservation):
   Preserves full unimodal representation for pure-edema (e.g. case0061) or pure-scar cases.
4. Dual-Resolution Strip + Local Anatomy Envelope:
   Combines global strip pooling with local myocardial convolution to attenuate extracardiac false positives.
"""
from __future__ import annotations

import torch
from torch import nn


class SpatialSkipGateFusion(nn.Module):
    """Multi-Scale Spatial-Anatomy Skip Gating:
    Uses CINE myocardium structure with a 3x3 depthwise-separable convolution
    to capture myocardial wall context and softly gate PSIR and T2w background noise,
    followed by 3x3 fusion and Squeeze-and-Excitation (SE) channel recalibration.
    """

    def __init__(self, channels: int):
        super().__init__()
        reduced = max(channels // 4, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        se_reduced = max(channels // 8, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, se_reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        g = self.gate(cine)
        # Calibrated residual gating: 20% minimum floor prevents erasing adjacent subtle pathology
        # while attenuating distant extracardiac noise by 80%
        calibrated_g = 0.2 + 0.8 * g
        fused = self.fusion(
            torch.cat([cine, psir * calibrated_g, t2w * calibrated_g], dim=1)
        )
        return fused * self.se(fused)


class M2MaxV4_Fusion(nn.Module):
    """Unimodal-Anchored Dual-Domain Cooperative Co-Attention with Calibrated Anatomy Envelope."""

    def __init__(self, in_channels: int = 1024, out_channels: int = 512, num_heads: int = 8):
        super().__init__()
        if in_channels % num_heads:
            raise ValueError("in_channels must be divisible by num_heads")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        # 1. Dual-Resolution Strip + Local Anatomy Envelope from CINE
        hidden = max(in_channels // 4, 16)
        self.strip_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.envelope_gate = nn.Sequential(
            nn.Conv2d(hidden * 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # 2. Multi-Head Cross-Attention modules
        # CINE queries Scar (LGE) and Edema (T2w)
        self.mha_cine_scar = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.mha_cine_edema = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)

        # Mutual Pathology Co-Attention
        self.mha_edema_to_scar = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.mha_scar_to_edema = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)

        # 3. Modality-Anchored Fusion (attn_c + attn_mutual + raw_modality) -> 3 * in_channels
        self.fuse_scar = nn.Sequential(
            nn.Conv2d(3 * in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse_edema = nn.Sequential(
            nn.Conv2d(3 * in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # 4. Dual-Domain (Spatial + Channel) Pathology Disambiguation
        self.joint_patho = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Dual-domain gates for Scar:
        ch_reduced = max(in_channels // 8, 16)
        self.scar_channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, ch_reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_reduced, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scar_spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Dual-domain gates for Edema:
        self.edema_channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, ch_reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_reduced, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.edema_spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # 5. Final projection into decoder dimension
        self.proj = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = cine.shape

        # Step 1: Dual-Resolution Strip + Local Anatomy Envelope from CINE
        strip_h = cine.mean(dim=3, keepdim=True)  # (B, C, H, 1)
        strip_w = cine.mean(dim=2, keepdim=True)  # (B, C, 1, W)
        strip_feat = self.strip_conv(strip_h + strip_w)  # (B, hidden, H, W)
        local_feat = self.local_conv(cine)  # (B, hidden, H, W)
        anatomy_gate = self.envelope_gate(torch.cat([strip_feat, local_feat], dim=1))  # (B, 1, H, W)

        # Step 2: Multi-head cross-attention tokens
        q_cine = cine.flatten(2).transpose(1, 2)
        k_scar = psir.flatten(2).transpose(1, 2)
        k_edema = t2w.flatten(2).transpose(1, 2)

        # (a) CINE-guided attention
        attn_c_scar, _ = self.mha_cine_scar(q_cine, k_scar, k_scar, need_weights=False)
        attn_c_edema, _ = self.mha_cine_edema(q_cine, k_edema, k_edema, need_weights=False)

        # (b) Mutual pathology co-attention
        attn_e_to_s, _ = self.mha_edema_to_scar(k_edema, k_scar, k_scar, need_weights=False)
        attn_s_to_e, _ = self.mha_scar_to_edema(k_scar, k_edema, k_edema, need_weights=False)

        # Reshape to spatial
        attn_c_scar = attn_c_scar.transpose(1, 2).reshape(batch, channels, height, width)
        attn_c_edema = attn_c_edema.transpose(1, 2).reshape(batch, channels, height, width)
        attn_e_to_s = attn_e_to_s.transpose(1, 2).reshape(batch, channels, height, width)
        attn_s_to_e = attn_s_to_e.transpose(1, 2).reshape(batch, channels, height, width)

        # Step 3: Modality-Anchored Fusion (attentions + raw unimodal skip)
        # Guarantees that pure-edema or pure-scar cases are NEVER starved
        scar_feat = self.fuse_scar(torch.cat([attn_c_scar, attn_s_to_e, psir], dim=1))
        edema_feat = self.fuse_edema(torch.cat([attn_c_edema, attn_e_to_s, t2w], dim=1))

        # Step 4: Dual-Domain (Spatial + Channel) Pathology Disambiguation
        joint = self.joint_patho(torch.cat([scar_feat, edema_feat], dim=1))
        
        # Dual gating: channel gate selects pathology features, spatial gate localizes them
        g_scar = self.scar_channel_gate(joint) * self.scar_spatial_gate(joint)
        g_edema = self.edema_channel_gate(joint) * self.edema_spatial_gate(joint)

        refined_scar = scar_feat + joint * g_scar
        refined_edema = edema_feat + joint * g_edema

        # Step 5: Calibrated anatomical envelope confinement (0.1 background floor + 0.9 gate)
        # Strongly attenuates extracardiac noise to suppress distant false positives and reduce HD95
        attenuation = 0.1 + 0.9 * anatomy_gate
        gated_scar = refined_scar * attenuation
        gated_edema = refined_edema * attenuation

        # Step 6: Final projection to decoder channel dimension
        return self.proj(torch.cat([cine, gated_scar, gated_edema], dim=1))
