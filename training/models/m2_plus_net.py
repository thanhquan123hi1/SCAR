"""M2-Plus: three ResNetV2 encoders, SSPANet and decoupled pathology attention."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ml_collections import ConfigDict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from training.models.backbones.resnet_v2 import ResNetV2, SharedResNetV2
from training.models.modules.decoder import DecoderCup, SegmentationHead
from training.models.modules.fusion import Fusion_Embed, SSPA_AttentionGate
from training.models.modules.m2_plus import (
    M2Plus_Fusion, SpatialAwareCrossAttention, LegacyM2PlusFusion,
    LEGACY_FUSION, SPATIAL_FUSION, FUSION_VERSIONS,
)
from training.models.modules.sspanet import (
    DEFAULT_SSPA_GATE,
    LEGACY_SSPA_GATE,
    OPTIMIZED_SSPA_GATE,
    SSPANet_Block,
    OptimizedSSPANet,
)


def get_lightweight_config() -> ConfigDict:
    """Production Lightweight M2-Plus configuration (< 15.0M parameters)."""
    config = ConfigDict()
    config.resnet = ConfigDict({"num_layers": (3, 4, 9), "width_factor": 1.0})
    config.transformer = ConfigDict({"dropout_rate": 0.0})
    config.decoder_channels = (128, 64, 32, 32)
    config.skip_channels = [512, 256, 64, 0]
    config.fused_channels = 512
    config.latent_dim = 256
    config.fusion_version = SPATIAL_FUSION
    config.use_pos_enc = True
    config.n_skip = 3
    config.n_classes = 4
    config.ablation = "LIGHTWEIGHT_M2_PLUS"
    config.cross_attention_heads = 8
    config.sspanet_reduction = 16
    config.sspanet_gate = OPTIMIZED_SSPA_GATE
    config.gradient_checkpointing = False
    config.classifier = "seg"
    config.activation = "logits"
    config.pretrained_path = None
    config.resnet_pretrained_path = None
    config.patches = ConfigDict({"size": (16, 16), "grid": (8, 8)})
    return config


def get_lightweight_testing() -> ConfigDict:
    """Testing configuration for LightweightM2PlusNet on CPU."""
    config = get_lightweight_config()
    config.resnet.num_layers = (1, 1, 1)
    config.resnet.width_factor = 0.5
    config.decoder_channels = (64, 32, 16, 16)
    config.skip_channels = [256, 128, 32, 0]
    config.fused_channels = 256
    return config


def get_config() -> ConfigDict:
    """Production M2-Plus configuration; legacy fields preserve checkpoint compatibility."""
    config = ConfigDict()
    config.resnet = ConfigDict({"num_layers": (3, 4, 9), "width_factor": 1.0})
    config.transformer = ConfigDict({"dropout_rate": 0.1})
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 0]
    config.fused_channels = 512
    config.latent_dim = 256
    config.fusion_version = SPATIAL_FUSION
    config.use_pos_enc = True
    config.n_skip = 3
    config.n_classes = 4
    config.ablation = "M2-PLUS"
    config.cross_attention_heads = 8
    config.sspanet_reduction = 16
    config.sspanet_gate = DEFAULT_SSPA_GATE
    config.gradient_checkpointing = False
    config.classifier = "seg"
    config.activation = "logits"
    config.pretrained_path = None
    config.resnet_pretrained_path = None
    config.patches = ConfigDict({"size": (16, 16), "grid": (8, 8)})
    return config


def get_testing() -> ConfigDict:
    """Small, structurally equivalent config for CPU tests, not experiments."""
    config = get_config()
    config.resnet.num_layers = (1, 1, 1)
    config.resnet.width_factor = 0.5
    config.decoder_channels = (64, 32, 16, 8)
    config.skip_channels = [256, 128, 32, 0]
    config.fused_channels = 64
    config.transformer.dropout_rate = 0.0
    return config


class _Embeddings(nn.Module):
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


class _Encoder(nn.Module):
    """Compatibility wrapper preserving legacy submodule hierarchy."""

    def __init__(self, config: ConfigDict):
        super().__init__()
        self.embeddings = _Embeddings(config)

    def forward(self, image: torch.Tensor):
        return self.embeddings(image)


class M2PlusNet(nn.Module):
    """Aligned CINE, PSIR/LGE and T2W images to four canonical-class logits.

    The transformer1/2/3 attribute names retain existing M2-Plus state-dict keys;
    these wrappers contain convolutional ResNetV2 encoders.
    """

    def __new__(cls, *args, **kwargs):
        config = kwargs.get("config", args[0] if len(args) > 0 else None)
        ablation = kwargs.get("ablation", None)
        is_lightweight = False
        if ablation is not None and str(ablation).upper() in ("LIGHTWEIGHT_M2_PLUS", "LIGHTWEIGHT"):
            is_lightweight = True
        elif config is not None:
            if hasattr(config, "ablation") and str(config.ablation).upper() in ("LIGHTWEIGHT_M2_PLUS", "LIGHTWEIGHT"):
                is_lightweight = True
            elif isinstance(config, dict) and str(config.get("ablation", "")).upper() in ("LIGHTWEIGHT_M2_PLUS", "LIGHTWEIGHT"):
                is_lightweight = True
        if is_lightweight and cls is M2PlusNet:
            return super().__new__(LightweightM2PlusNet)
        return super().__new__(cls)

    def __init__(
        self,
        config: ConfigDict | dict[str, Any] | None = None,
        img_size: int = 128,
        num_classes: int | None = None,
        zero_head: bool = False,
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

        # Checkpoints predating gate metadata must retain their original math.
        for key, value in (("gradient_checkpointing", False), ("fused_channels", 512),
                           ("cross_attention_heads", 8), ("sspanet_gate", LEGACY_SSPA_GATE),
                           ("fusion_version", LEGACY_FUSION), ("sspanet_reduction", 16)):
            if key not in self.config:
                self.config[key] = value

        configured = self.config.get("ablation", "M2-PLUS")
        for selected in (configured, ablation or configured):
            if str(selected).upper() not in ("M2-PLUS", "LIGHTWEIGHT_M2_PLUS", "LIGHTWEIGHT"):
                raise ValueError(f"Only M2-Plus is supported; received {selected!r}")
        self.ablation = "M2-PLUS"
        self.config.ablation = self.ablation
        self.config.architecture = "m2_plus"

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
        if self.config.fusion_version not in FUSION_VERSIONS:
            raise ValueError(f"Unknown fusion version: {self.config.fusion_version}")
        if self.config.fusion_version == LEGACY_FUSION:
            self.config.latent_dim = channels
            self.config.use_pos_enc = False
        else:
            self.config.latent_dim = self.config.get("latent_dim", min(256, channels))
            self.config.use_pos_enc = self.config.get("use_pos_enc", True)
        self.transformer1 = _Encoder(self.config)
        self.transformer2 = _Encoder(self.config)
        self.transformer3 = _Encoder(self.config)

        self.sspanet_cine = SSPANet_Block(channels, gate_mode=self.config.sspanet_gate, reduction_ratio=self.config.sspanet_reduction)
        self.sspanet_psir = SSPANet_Block(channels, gate_mode=self.config.sspanet_gate, reduction_ratio=self.config.sspanet_reduction)
        self.sspanet_t2w = SSPANet_Block(channels, gate_mode=self.config.sspanet_gate, reduction_ratio=self.config.sspanet_reduction)

        if self.config.fusion_version == LEGACY_FUSION:
            self.cross_fusion = LegacyM2PlusFusion(channels, self.config.fused_channels, self.config.cross_attention_heads)
        else:
            self.cross_fusion = M2Plus_Fusion(
                in_channels=channels, latent_dim=self.config.latent_dim,
                out_channels=self.config.fused_channels, num_heads=self.config.cross_attention_heads,
                use_pos_enc=self.config.use_pos_enc,
            )
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
        dec_out = self.decoder(fused, skips)
        return self.segmentation_head(dec_out)

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


class SlimDecoderBlock(nn.Module):
    """Progressive upsampling block with depthwise-separable skip fusion.

    1. Bilinear upsampling 2x
    2. Input projection: in_channels -> out_channels (1x1 Conv + GroupNorm + ReLU)
    3. Skip fusion (if skip provided):
       - Skip projection: skip_channels -> out_channels // 2 (1x1 Conv + GroupNorm + ReLU)
       - Concatenation: out_channels + out_channels // 2
       - Depthwise Separable Conv: 3x3 depthwise conv + 1x1 pointwise conv + GroupNorm + ReLU
    """

    def __init__(self, in_channels: int, skip_channels: int = 0, out_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels

        num_gn = min(8, out_channels)
        while out_channels % num_gn != 0 and num_gn > 1:
            num_gn -= 1

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=num_gn, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )
        if skip_channels > 0:
            skip_mid = max(out_channels // 2, 16)
            num_gn_skip = min(8, skip_mid)
            while skip_mid % num_gn_skip != 0 and num_gn_skip > 1:
                num_gn_skip -= 1

            self.proj_skip = nn.Sequential(
                nn.Conv2d(skip_channels, skip_mid, 1, bias=False),
                nn.GroupNorm(num_groups=num_gn_skip, num_channels=skip_mid),
                nn.ReLU(inplace=True),
            )
            fuse_in = out_channels + skip_mid
            self.fuse = nn.Sequential(
                nn.Conv2d(fuse_in, fuse_in, 3, padding=1, groups=fuse_in, bias=False),
                nn.Conv2d(fuse_in, out_channels, 1, bias=False),
                nn.GroupNorm(num_groups=num_gn, num_channels=out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.proj_skip = None
            self.fuse = None

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.proj_in(x)
        if skip is not None and self.proj_skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
            s = self.proj_skip(skip)
            x = self.fuse(torch.cat([x, s], dim=1))
        return x


class LightweightM2PlusNet(M2PlusNet):
    """Lightweight M2-Plus multi-sequence cardiac magnetic resonance segmentation model.

    Adheres strictly to PROJECT.md §1, §4.4 and research integrity constraints:
    - Shared-Weight ResNetV2-50 backbone with independent 1-channel stems and FiLM adapters (~11.91M params)
    - 3 Bottleneck OptimizedSSPANet blocks (Z-Pool, Depthwise Strip Pooling, 1st/2nd-order Style Pooling, ~1.38M params)
    - Spatial-Aware Decoupled Cross-Attention (256 latent dim, 2D Sinusoidal Positional Encoding, ~1.18M params)
    - Multi-Level SSPA Attention Gates on skip connections (<0.02M params)
    - Slim Progressive Decoder with depthwise skip fusion and segmentation head (~0.15M params)
    - Total Model Parameter Budget: strictly < 15,000,000 parameters (~14.67M params, ~79.8% reduction from 72.54M).
    """

    def __init__(
        self,
        config: ConfigDict | dict[str, Any] | None = None,
        img_size: int = 128,
        num_classes: int | None = None,
        zero_head: bool = False,
        ablation: str | None = None,
        pretrained_path: str | Path | None = None,
        trunk_layers: tuple[int, int, int] | None = None,
        width: float | None = None,
    ):
        nn.Module.__init__(self)

        if config is None:
            raw_config = get_lightweight_config()
        elif isinstance(config, dict) and not isinstance(config, ConfigDict):
            raw_config = ConfigDict(config)
        else:
            raw_config = config
        self.config = deepcopy(raw_config)

        if trunk_layers is not None:
            if not hasattr(self.config, "resnet"):
                self.config.resnet = ConfigDict()
            self.config.resnet.num_layers = trunk_layers
        if width is not None:
            if not hasattr(self.config, "resnet"):
                self.config.resnet = ConfigDict()
            self.config.resnet.width_factor = width
        if num_classes is not None:
            self.config.n_classes = int(num_classes)
        if ablation is not None:
            self.config.ablation = ablation

        self.num_classes = getattr(self.config, "n_classes", 4)
        self.img_size = img_size
        self.ablation = "LIGHTWEIGHT_M2_PLUS"
        self.config.ablation = self.ablation
        self.config.architecture = "lightweight_m2_plus"
        if self.num_classes < 2:
            raise ValueError("At least two segmentation classes are required")

        resnet_cfg = getattr(self.config, "resnet", ConfigDict())
        num_layers = getattr(resnet_cfg, "num_layers", (3, 4, 9))
        width_factor = getattr(resnet_cfg, "width_factor", 1.0)
        grad_ckpt = getattr(self.config, "gradient_checkpointing", False)

        w = int(64 * width_factor)
        bottleneck_channels = 16 * w
        fused_channels = getattr(self.config, "fused_channels", 512 if width_factor == 1.0 else 256)
        latent_dim = getattr(self.config, "latent_dim", min(256, bottleneck_channels))
        cross_heads = getattr(self.config, "cross_attention_heads", 8)
        sspa_reduction = getattr(self.config, "sspanet_reduction", 16)
        if self.config.get("fusion_version", SPATIAL_FUSION) != SPATIAL_FUSION:
            raise ValueError("Lightweight M2-Plus requires spatial_v2 fusion")
        if self.config.get("sspanet_gate", OPTIMIZED_SSPA_GATE) != OPTIMIZED_SSPA_GATE:
            raise ValueError("Lightweight M2-Plus requires optimized SSPANet")
        self.config.fusion_version = SPATIAL_FUSION
        self.config.latent_dim = latent_dim
        self.config.use_pos_enc = self.config.get("use_pos_enc", True)
        self.config.sspanet_gate = OPTIMIZED_SSPA_GATE
        self.config.sspanet_reduction = sspa_reduction

        # 1. Shared ResNetV2 backbone with 3 independent 1-channel stems and FiLM adapters
        self.encoder = SharedResNetV2(
            block_units=num_layers,
            width_factor=width_factor,
            gradient_checkpointing=grad_ckpt,
        )

        # 2. OptimizedSSPANet bottleneck blocks for CINE, PSIR/LGE, and T2W
        self.sspanet_cine = OptimizedSSPANet(bottleneck_channels, reduction_ratio=sspa_reduction, mode="residual")
        self.sspanet_psir = OptimizedSSPANet(bottleneck_channels, reduction_ratio=sspa_reduction, mode="residual")
        self.sspanet_t2w = OptimizedSSPANet(bottleneck_channels, reduction_ratio=sspa_reduction, mode="residual")

        # 3. Spatial-Aware Decoupled Cross-Attention (latent dim 256, 2D Sinusoidal Positional Encoding)
        self.cross_attention = SpatialAwareCrossAttention(
            in_channels=bottleneck_channels,
            latent_dim=latent_dim,
            out_channels=fused_channels,
            num_heads=cross_heads,
            use_pos_enc=self.config.use_pos_enc,
        )
        self.cross_fusion = self.cross_attention

        # 4. Skip Attention Gates (SSPA_AttentionGate) filtering thoracic noise on skips
        skip0_ch = 8 * w
        skip1_ch = 4 * w
        skip2_ch = w
        dec_ch = getattr(self.config, "decoder_channels", (128, 64, 32, 32) if width_factor == 1.0 else (64, 32, 16, 16))

        self.gate_skip0 = SSPA_AttentionGate(
            in_channels=skip0_ch,
            gate_channels=fused_channels,
            inter_channels=max(skip0_ch // 16, 16),
        )
        self.gate_skip1 = SSPA_AttentionGate(
            in_channels=skip1_ch,
            gate_channels=dec_ch[0],
            inter_channels=max(skip1_ch // 16, 16),
        )
        self.gate_skip2 = SSPA_AttentionGate(
            in_channels=skip2_ch,
            gate_channels=dec_ch[1],
            inter_channels=max(skip2_ch // 16, 16),
        )

        # 5. Progressive Slim Decoder
        self.dec1 = SlimDecoderBlock(in_channels=fused_channels, skip_channels=skip0_ch, out_channels=dec_ch[0])
        self.dec2 = SlimDecoderBlock(in_channels=dec_ch[0], skip_channels=skip1_ch, out_channels=dec_ch[1])
        self.dec3 = SlimDecoderBlock(in_channels=dec_ch[1], skip_channels=skip2_ch, out_channels=dec_ch[2])
        self.dec4 = SlimDecoderBlock(in_channels=dec_ch[2], skip_channels=0, out_channels=dec_ch[3])
        self.decoder = nn.ModuleList([self.dec1, self.dec2, self.dec3, self.dec4])

        # 6. Final Segmentation Head
        self.head = nn.Conv2d(dec_ch[3], self.num_classes, kernel_size=1)
        self.segmentation_head = self.head

        if zero_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        if pretrained_path is not None:
            self.load_pretrained_encoders(pretrained_path)

    @property
    def shared_trunk(self) -> SharedResNetV2:
        return self.encoder

    @property
    def stem_cine(self) -> nn.Module:
        return self.encoder.stem_cine

    @property
    def stem_psir(self) -> nn.Module:
        return self.encoder.stem_psir

    @property
    def stem_t2w(self) -> nn.Module:
        return self.encoder.stem_t2w

    @property
    def encoder_cine(self) -> SharedResNetV2:
        return self.encoder

    @property
    def encoder_psir(self) -> SharedResNetV2:
        return self.encoder

    @property
    def encoder_t2w(self) -> SharedResNetV2:
        return self.encoder

    def _forward_encoder_modality(self, x: torch.Tensor, stem: nn.Module | str):
        if isinstance(stem, str):
            return self.encoder.forward_modality(x, stem)
        for name in ("cine", "psir", "t2w"):
            if getattr(self.encoder, f"stem_{name}", None) is stem:
                return self.encoder.forward_modality(x, name)
        return self.encoder.forward_modality(x, "cine")

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
        target_h, target_w = cine.shape[2:]

        # 1. Encoders: SharedResNetV2 with independent 1-ch stems and FiLM adapters
        feat_cine, skips_cine = self.encoder.forward_modality(cine, "cine")
        feat_psir, skips_psir = self.encoder.forward_modality(psir, "psir")
        feat_t2w, skips_t2w = self.encoder.forward_modality(t2w, "t2w")

        # 2. Bottlenecks: 3 OptimizedSSPANet blocks
        att_cine = self.sspanet_cine(feat_cine)
        att_psir = self.sspanet_psir(feat_psir)
        att_t2w = self.sspanet_t2w(feat_t2w)

        # 3. Spatial-Aware Decoupled Cross-Attention
        fused = self.cross_attention(att_cine, att_psir, att_t2w)

        # 4. Multi-modal skip connections: combine CINE anatomy with PSIR scar and T2w edema
        skip0 = (skips_cine[0] + skips_psir[0] + skips_t2w[0]) / 3.0
        skip1 = (skips_cine[1] + skips_psir[1] + skips_t2w[1]) / 3.0
        skip2 = (skips_cine[2] + skips_psir[2] + skips_t2w[2]) / 3.0

        # 5. Progressive Slim Decoder with SSPA Skip Attention Gates (thoracic noise filtering)
        # Stage 1: H/16 -> H/8, gated skip0
        gated_skip0 = self.gate_skip0(skip0, g=fused)
        d1 = self.dec1(fused, gated_skip0)

        # Stage 2: H/8 -> H/4, gated skip1
        gated_skip1 = self.gate_skip1(skip1, g=d1)
        d2 = self.dec2(d1, gated_skip1)

        # Stage 3: H/4 -> H/2, gated skip2
        gated_skip2 = self.gate_skip2(skip2, g=d2)
        d3 = self.dec3(d2, gated_skip2)

        # Stage 4: H/2 -> H
        d4 = self.dec4(d3)
        if d4.shape[2:] != (target_h, target_w):
            d4 = F.interpolate(d4, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # 6. Final Segmentation Head
        logits = self.head(d4)
        return logits

    def load_pretrained_encoders(self, path: str | Path):
        self.encoder.load_from_transunet(path)

    def load_from(self, weights):
        self.encoder.load_from(weights)


__all__ = [
    "M2PlusNet",
    "LightweightM2PlusNet",
    "SlimDecoderBlock",
    "get_config",
    "get_testing",
    "get_lightweight_config",
    "get_lightweight_testing",
]
