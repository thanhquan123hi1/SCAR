"""Modality fusion modules for multi-modal feature representations."""
from __future__ import annotations

import torch
from torch import nn


class ConcatFusion(nn.Module):
    def __init__(self, in_channels=1024, out_channels=512):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cine, psir, t2w):
        return self.fusion_conv(torch.cat((cine, psir, t2w), dim=1))


class Fusion_Embed(ConcatFusion):
    """Concatenate modalities, then project to the original skip width."""

    def __init__(self, embed_dim):
        super().__init__(embed_dim, embed_dim)
