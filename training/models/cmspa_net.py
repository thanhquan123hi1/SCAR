"""Prompt-free multi-modal CMR segmentation network (CMSPA-Net).

Supports M2 and M3 ablations.
CMSPA-Net uses three independent ResNetV2 encoders (CINE, PSIR/LGE, T2W),
SSPANet spatial and channel attention, cross-modal strip pathology attention (CMSPA) or cross-attention,
and a single decoder cascade with skip fusion.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ml_collections import ConfigDict
import numpy as np
import torch
from torch import nn

from training.models.backbones.resnet_v2 import ResNetV2
from training.models.modules.cmspa import CMSPA_Fusion
from training.models.modules.decoder import DecoderCup, SegmentationHead
from training.models.modules.fusion import CrossAttention_Fusion, Fusion_Embed
from training.models.modules.sspanet import SSPANet_Block


def get_r50_b16_config() -> ConfigDict:
    """Production configuration for R50-B16 based CMSPA-Net."""
    config = ConfigDict()
    config.resnet = ConfigDict({"num_layers": (3, 4, 9), "width_factor": 1.0})
    config.transformer = ConfigDict({"dropout_rate": 0.1})
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 0]
    config.fused_channels = 512
    config.n_skip = 3
    config.n_classes = 4
    config.ablation = "M3"
    config.cross_attention_heads = 8
    config.gradient_checkpointing = False
    config.classifier = "seg"
    config.activation = "logits"
    config.pretrained_path = None
    config.resnet_pretrained_path = None
    config.patches = ConfigDict({"size": (16, 16), "grid": (8, 8)})
    return config


def get_config(ablation: str = "M3") -> ConfigDict:
    """Return a fresh production configuration."""
    if ablation.upper() not in {"M2", "M3"}:
        raise ValueError(f"Unknown ablation {ablation!r}; expected M2 or M3")
    config = get_r50_b16_config()
    config.ablation = ablation.upper()
    return config


def get_testing() -> ConfigDict:
    """Small, structurally equivalent config for CPU tests, not experiments."""
    config = get_r50_b16_config()
    config.resnet.num_layers = (1, 1, 1)
    config.resnet.width_factor = 0.5
    config.decoder_channels = (64, 32, 16, 8)
    config.skip_channels = [256, 128, 32, 0]
    config.fused_channels = 64
    config.transformer.dropout_rate = 0.0
    return config


def get_r50_l16_config() -> ConfigDict:
    """Legacy name; prompt-free encoder is identical to R50-ViT-B_16."""
    return get_r50_b16_config()


CONFIGS: dict[str, ConfigDict] = {
    "R50-ViT-B_16": get_r50_b16_config(),
    "R50-ViT-L_16": get_r50_l16_config(),
    "testing": get_testing(),
}


class Embeddings(nn.Module):
    """Convolutional encoder wrapper preserving legacy attribute hierarchy."""

    def __init__(self, config: ConfigDict):
        super().__init__()
        self.hybrid_model = ResNetV2(
            config.resnet.num_layers,
            config.resnet.width_factor,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        self.dropout = nn.Dropout(config.transformer.dropout_rate)

    def forward(self, image: torch.Tensor):
        feature, skips = self.hybrid_model(image)
        return self.dropout(feature), skips


class Transformer(nn.Module):
    """Compatibility wrapper preserving legacy submodule hierarchy."""

    def __init__(self, config: ConfigDict, img_size: int | None = None, vis: bool = False):
        super().__init__()
        self.embeddings = Embeddings(config)

    def forward(self, image: torch.Tensor):
        return self.embeddings(image)


class CMSPANet(nn.Module):
    """Three independent ResNetV2 encoders and one segmentation decoder.

    Ablations:
      M2: SSPANet + cross-attention
      M3: SSPANet + CMSPA (proposed)

    Inputs: aligned floating (B,1,H,W) or (B,3,H,W) tensors in order CINE, PSIR/LGE, T2W.
    Returns: raw logits (B,num_classes,H,W) in train AND eval modes.
    """

    def __init__(
        self,
        config: ConfigDict | dict[str, Any] | None = None,
        img_size: int = 128,
        num_classes: int | None = None,
        zero_head: bool = False,
        vis: bool = False,
        ablation: str | None = None,
        pretrained_path: str | Path | None = None,
    ):
        super().__init__()
        if config is None:
            raw_config = get_config()
        elif isinstance(config, dict) and not isinstance(config, ConfigDict):
            raw_config = ConfigDict(config)
        else:
            raw_config = config
        self.config = deepcopy(raw_config)

        for key, value in (("gradient_checkpointing", False), ("fused_channels", 512),
                           ("cross_attention_heads", 8)):
            if key not in self.config:
                self.config[key] = value

        self.ablation = (ablation or self.config.get("ablation", "M3")).upper()
        if self.ablation not in {"M2", "M3"}:
            raise ValueError(f"Unknown ablation: {self.ablation!r}; expected M2 or M3")
        self.config.ablation = self.ablation

        if num_classes is not None:
            self.config.n_classes = int(num_classes)
        self.num_classes = self.config.n_classes
        if self.num_classes < 2:
            raise ValueError("At least two segmentation classes are required")
        if self.config.n_skip not in range(4):
            raise ValueError("n_skip must be between 0 and 3")
        if len(self.config.decoder_channels) != 4 or len(self.config.skip_channels) != 4:
            raise ValueError("Exactly four decoder widths and skip widths are required")
        self.img_size = img_size

        width = int(64 * self.config.resnet.width_factor)
        expected_skips = [8 * width, 4 * width, width]
        if list(self.config.skip_channels[:self.config.n_skip]) != expected_skips[:self.config.n_skip]:
            raise ValueError(f"Skip channels must match encoder widths {expected_skips}")

        channels = 16 * width
        self.transformer1 = Transformer(self.config)
        self.transformer2 = Transformer(self.config)
        self.transformer3 = Transformer(self.config)

        self.sspanet_cine = SSPANet_Block(channels)
        self.sspanet_psir = SSPANet_Block(channels)
        self.sspanet_t2w = SSPANet_Block(channels)

        if self.ablation == "M3":
            self.cross_fusion = CMSPA_Fusion(channels, self.config.fused_channels)
        elif self.ablation == "M2":
            self.cross_fusion = CrossAttention_Fusion(
                channels, self.config.fused_channels, self.config.cross_attention_heads
            )
        else:
            raise ValueError(f"Unsupported ablation: {self.ablation!r}")

        self.feature_fusion = nn.ModuleList(
            Fusion_Embed(c) for c in expected_skips[:self.config.n_skip]
        )
        self.decoder = DecoderCup(self.config)
        self.segmentation_head = SegmentationHead(
            self.config.decoder_channels[-1], self.num_classes
        )

        if zero_head:
            nn.init.zeros_(self.segmentation_head[0].weight)
            nn.init.zeros_(self.segmentation_head[0].bias)

        if pretrained_path is not None:
            self.load_pretrained_encoders(pretrained_path)

        self._register_load_state_dict_pre_hook(self._remap_state_dict_hook)

    @property
    def encoder_cine(self) -> ResNetV2:
        return self.transformer1.embeddings.hybrid_model

    @property
    def encoder_psir(self) -> ResNetV2:
        return self.transformer2.embeddings.hybrid_model

    @property
    def encoder_t2w(self) -> ResNetV2:
        return self.transformer3.embeddings.hybrid_model

    @staticmethod
    def _remap_state_dict_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Bidirectional compatibility hook translating modern alias keys to internal keys."""
        remap = [
            ("encoder_cine.", "transformer1.embeddings.hybrid_model."),
            ("encoder_psir.", "transformer2.embeddings.hybrid_model."),
            ("encoder_t2w.", "transformer3.embeddings.hybrid_model."),
        ]
        keys = list(state_dict.keys())
        for k in keys:
            for clean_p, legacy_p in remap:
                full_clean = prefix + clean_p
                if k.startswith(full_clean):
                    new_k = prefix + legacy_p + k[len(full_clean):]
                    state_dict[new_k] = state_dict.pop(k)

    def _validate_inputs(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor):
        for name, image in (("CINE", cine), ("PSIR", psir), ("T2W", t2w)):
            if image.ndim != 4 or image.shape[1] not in (1, 3):
                raise ValueError(f"{name} must have shape (B,1,H,W) or (B,3,H,W)")
            if not image.is_floating_point():
                raise TypeError(f"{name} input must be floating point")
            if image.shape[0] != cine.shape[0] or image.shape[2:] != cine.shape[2:]:
                raise ValueError("All modalities must have aligned batch and spatial dimensions")
            if image.device != cine.device or image.dtype != cine.dtype:
                raise ValueError("All modalities must have the same device and dtype")
        if any(size < 32 or size % 16 for size in cine.shape[2:]):
            raise ValueError("Input H and W must be >=32 and divisible by 16")

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(cine, psir, t2w)
        images = [image.repeat(1, 3, 1, 1) if image.shape[1] == 1 else image
                  for image in (cine, psir, t2w)]
        cine_f, cine_skips = self.transformer1(images[0])
        psir_f, psir_skips = self.transformer2(images[1])
        t2w_f, t2w_skips = self.transformer3(images[2])

        fused = self.cross_fusion(
            self.sspanet_cine(cine_f),
            self.sspanet_psir(psir_f),
            self.sspanet_t2w(t2w_f),
        )
        skips = [
            fusion(cine_skips[i], psir_skips[i], t2w_skips[i])
            for i, fusion in enumerate(self.feature_fusion)
        ]
        return self.segmentation_head(self.decoder(fused, skips))

    def load_pretrained_encoders(self, path: str | Path):
        """Explicitly load matching JAX/TransUNet ResNet weights into all branches."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Encoder checkpoint does not exist: {path}")
        with np.load(path, allow_pickle=False) as weights:
            self.load_from(weights)

    def load_from(self, weights):
        """Load convolutional encoders; no text, ViT or decoder weights."""
        encoders = [module.embeddings.hybrid_model
                    for module in (self.transformer1, self.transformer2, self.transformer3)]
        # Fail on incompatible NPZ before mutating any branch.
        encoders[0].validate_pretrained(weights)
        for encoder in encoders:
            encoder.load_from(weights)


VisionTransformer = CMSPANet
