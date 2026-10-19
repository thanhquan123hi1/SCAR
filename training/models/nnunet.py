"""nnU-Net 2D PlainConvUNet wrapper for SCAR early-fusion multi-modal segmentation."""
from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn

# Ensure dynamic_network_architectures package is importable
_MODELS_DIR = str(Path(__file__).parent)
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

logger = logging.getLogger(__name__)


class NNUNet2D(nn.Module):
    """Early-fusion 2D nnU-Net (PlainConvUNet) wrapper compatible with SCAR contracts.

    Accepts 3 modalities (Cine, LGE, T2w) as individual tensors (B, 1, H, W)
    or 1 concatenated tensor (B, 3, H, W), concatenates them along the channel
    dimension, and produces raw logits of shape (B, 4, H, W).
    """

    def __init__(
        self,
        num_classes: int = 4,
        img_size: int = 128,
        features_per_stage: tuple[int, ...] = (32, 64, 128, 256, 512),
        **kwargs: Any,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        n_stages = len(features_per_stage)

        strides = [(1, 1)] + [(2, 2)] * (n_stages - 1)
        kernel_sizes = [(3, 3)] * n_stages
        n_conv_per_stage = [2] * n_stages
        n_conv_per_stage_decoder = [2] * (n_stages - 1)

        self.net = PlainConvUNet(
            input_channels=3,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv2d,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=nn.InstanceNorm2d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True, "negative_slope": 1e-2},
            deep_supervision=False,
            nonlin_first=False,
        )
        self.net.apply(InitWeights_He(1e-2))

    def forward(self, *images: torch.Tensor) -> torch.Tensor:
        """Forward pass supporting either 3 modality tensors or 1 pre-concatenated tensor."""
        if len(images) == 1:
            x = images[0]
        elif len(images) == 3:
            x = torch.cat(images, dim=1)
        else:
            raise ValueError(f"Expected 1 or 3 input tensors, got {len(images)}")
        return self.net(x)

    def forward_for_loss(self, *images: torch.Tensor) -> torch.Tensor:
        """Trainer hook: returns logits directly for loss calculation."""
        return self.forward(*images)

    def load_pretrained_encoders(self, pretrained_path: str | None = None) -> None:
        """No-op: nnU-Net is trained from scratch with He initialization."""
        if pretrained_path:
            logger.info("nnU-Net is trained from scratch; ignoring pretrained weights %s", pretrained_path)
