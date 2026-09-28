"""M2-Max: Bi-Pathology Interactive Cross-Attention with Anatomical Skip Gating.
Combines decoupled scar/edema attention with mutual competitive refinement
and anatomical CINE gating at multi-scale skip connections.
"""
from __future__ import annotations

import torch
from torch import nn


class SkipGateFusion(nn.Module):
    """Anatomical Skip Gating:
    Uses CINE myocardium structure to softly gate PSIR and T2w background noise,
    followed by Squeeze-and-Excitation (SE) channel recalibration.
    """

    def __init__(self, channels: int):
        super().__init__()
        reduced = max(channels // 4, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False),
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
        # Residual gating: preserves signal while suppressing distant artifacts
        fused = self.fusion(
            torch.cat([cine, psir * (0.5 + 0.5 * g), t2w * (0.5 + 0.5 * g)], dim=1)
        )
        return fused * self.se(fused)


class M2Max_Fusion(nn.Module):
    """Bi-Pathology Interactive Cross-Attention:
    1. Decoupled Cross-Attention:
       - CINE queries PSIR -> attn_scar
       - CINE queries T2w  -> attn_edema
    2. Bi-Pathology Interactive Refinement:
       - Joint pathology features: Conv3x3([attn_scar, attn_edema])
       - Competitive spatial allocation gate: Sigmoid(Conv1x1(joint))
       - Scar gets reinforced where scar gate is high; Edema where it is low.
    3. Final projection of (CINE + refined_scar + refined_edema) -> out_channels.
    """

    def __init__(self, in_channels: int = 1024, out_channels: int = 512, num_heads: int = 8):
        super().__init__()
        if in_channels % num_heads:
            raise ValueError("Channels must be divisible by num_heads")

        self.mha_scar = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.mha_edema = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)

        # Joint pathology modeling at the bottleneck
        self.joint_patho = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.comp_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Final projection into decoder dimension
        self.proj = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = cine.shape
        query = cine.flatten(2).transpose(1, 2)
        key_scar = psir.flatten(2).transpose(1, 2)
        key_edema = t2w.flatten(2).transpose(1, 2)

        # 1. Primary decoupled cross-attention
        attn_scar, _ = self.mha_scar(query, key_scar, key_scar, need_weights=False)
        attn_edema, _ = self.mha_edema(query, key_edema, key_edema, need_weights=False)

        attn_scar = attn_scar.transpose(1, 2).reshape(batch, channels, height, width)
        attn_edema = attn_edema.transpose(1, 2).reshape(batch, channels, height, width)

        # 2. Bi-pathology competitive refinement
        joint = self.joint_patho(torch.cat([attn_scar, attn_edema], dim=1))
        gate = self.comp_gate(joint)

        refined_scar = attn_scar + joint * gate
        refined_edema = attn_edema + joint * (1.0 - gate)

        # 3. Concatenate CINE anatomical prior with refined pathology representations
        return self.proj(torch.cat([cine, refined_scar, refined_edema], dim=1))
