"""M2-Pro: Infarct-Guided Cascade Cross-Attention for Myocardial Pathology.
Connects the Scar attention stream to condition the Edema attention stream.
"""
from __future__ import annotations

import torch
from torch import nn


class M2Pro_Fusion(nn.Module):
    """Infarct-Guided Cascade Cross-Attention:
    1. Stage 1: CINE queries attend to PSIR/LGE (Scar specialist) -> attn_scar
    2. Stage 2: CINE query conditioned with attn_scar queries T2W (Edema specialist)
       Because edema surrounds and co-localizes with the infarct core, conditioning the
       edema query with scar attention provides a strong spatial localization prior on T2w!
    3. Soft strip anatomy gating from CINE softly bounds the output within the myocardium.
    4. Projection of (CINE + attn_scar + attn_edema) -> out_channels.
    """

    def __init__(self, in_channels: int = 1024, out_channels: int = 512, num_heads: int = 8):
        super().__init__()
        if in_channels % num_heads:
            raise ValueError("Channels must be divisible by num_heads")

        self.mha_scar = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.mha_edema = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)

        # Condition projection: fuses CINE query with Scar attention to guide Edema attention
        self.edema_query_proj = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True),
        )

        # Soft strip anatomy gating from CINE
        hidden = max(in_channels // 4, 1)
        self.conv_strip = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

        # Final projection to decoder width
        self.proj = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = cine.shape

        # 1. Soft anatomy prior from CINE
        anatomy_gate = self.conv_strip(
            cine.mean(dim=3, keepdim=True) + cine.mean(dim=2, keepdim=True)
        )

        # 2. Stage 1: Scar cross-attention (CINE queries PSIR)
        query_cine = cine.flatten(2).transpose(1, 2)
        key_scar = psir.flatten(2).transpose(1, 2)
        attn_scar, _ = self.mha_scar(query_cine, key_scar, key_scar, need_weights=False)

        # 3. Stage 2: Infarct-Guided Edema cross-attention
        # Condition edema query with scar attention features: query_edema = proj([query_cine, attn_scar])
        guided_query = self.edema_query_proj(torch.cat((query_cine, attn_scar), dim=-1))
        key_edema = t2w.flatten(2).transpose(1, 2)
        attn_edema, _ = self.mha_edema(guided_query, key_edema, key_edema, need_weights=False)

        # Reshape back to spatial dimensions
        attn_scar = attn_scar.transpose(1, 2).reshape(batch, channels, height, width)
        attn_edema = attn_edema.transpose(1, 2).reshape(batch, channels, height, width)

        # Residual anatomy gating: guarantees gradients flow freely while attenuating outer noise
        gated_scar = attn_scar * (0.5 + 0.5 * anatomy_gate)
        gated_edema = attn_edema * (0.5 + 0.5 * anatomy_gate)

        # Project concatenated features into decoder width
        return self.proj(torch.cat((cine, gated_scar, gated_edema), dim=1))
