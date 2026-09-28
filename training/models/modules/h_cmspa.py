"""H-CMSPA: Hierarchical Cross-Modal Strip-Pathology Attention Fusion.
Combines CINE Strip Anatomy Guidance (boundary prior) with Decoupled Pathology Cross-Attention.
"""
from __future__ import annotations

import torch
from torch import nn

from training.models.modules.sspanet import _statistics_input


def channel_std(x, epsilon=1e-6):
    """Per-pixel channel population std; also defined when C=1."""
    return (_statistics_input(x).var(dim=1, keepdim=True, unbiased=False) + epsilon).sqrt()


class H_CMSPA_Fusion(nn.Module):
    """Hierarchical Cross-Modal Strip-Pathology Attention Fusion:
    1. CINE queries attend to PSIR/LGE (Scar specialist)
    2. CINE queries attend to T2W (Edema specialist)
    3. Strip Anatomy Guidance (CINE spatial prior) gates both attention branches,
       preventing pathology features from leaking outside the myocardial envelope.
    4. Style-based pathology contrast enhancement:
       Enhances subtle edema/scar features using channel standard deviation feedback.
    5. Fused projection to decoder channel dimension.
    """

    def __init__(self, in_channels: int = 1024, out_channels: int = 512, num_heads: int = 8):
        super().__init__()
        if in_channels % num_heads:
            raise ValueError("Channels must be divisible by num_heads")

        hidden = max(in_channels // 4, 1)
        # Strip Anatomy Gating from CINE
        self.conv_strip = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        # Pathology style feedback
        self.conv_patho = nn.Sequential(
            nn.Conv2d(1, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

        # Decoupled Multihead Cross-Attention
        self.mha_scar = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.mha_edema = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)

        # Projection of CINE + Gated_Scar + Gated_Edema
        self.proj = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = cine.shape

        # 1. Strip anatomy guidance from CINE (B, 1, H, W)
        anatomy_gate = self.conv_strip(
            cine.mean(dim=3, keepdim=True) + cine.mean(dim=2, keepdim=True)
        )

        # 2. Decoupled cross-attention
        query = cine.flatten(2).transpose(1, 2)
        key_scar = psir.flatten(2).transpose(1, 2)
        key_edema = t2w.flatten(2).transpose(1, 2)

        attn_scar, _ = self.mha_scar(query, key_scar, key_scar, need_weights=False)
        attn_edema, _ = self.mha_edema(query, key_edema, key_edema, need_weights=False)

        attn_scar = attn_scar.transpose(1, 2).reshape(batch, channels, height, width)
        attn_edema = attn_edema.transpose(1, 2).reshape(batch, channels, height, width)

        # 3. Apply Anatomical Gating: locks pathology features within the myocardial envelope
        gated_scar = attn_scar * anatomy_gate
        gated_edema = attn_edema * anatomy_gate

        # 4. Pathology style enhancement on CINE
        style = (channel_std(psir) + channel_std(t2w)).to(cine.dtype)
        pathology_gate = self.conv_patho(style)
        cine_enhanced = cine * (1.0 + pathology_gate)

        # 5. Project concatenated features
        return self.proj(torch.cat((cine_enhanced, gated_scar, gated_edema), dim=1))
