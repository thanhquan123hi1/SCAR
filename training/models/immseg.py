"""I-MMSeg: Multi-Modal Interaction Network with Pre-computed Text Embeddings.

Extracted and canonicalized from I_MMSeg official repository:
    https://github.com/thanhquan123hi1/SCAR.git
    Architecture: Triple-branch ResNetV2 hybrid encoder + Cross-modal Cosine Attention
                  + CBAM feature modulation + Dynamic forward prediction head.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict
from scipy import ndimage

from training.models.backbones.resnet_v2 import ResNetV2, np2th

TEXT_FEATURES_DIR = Path(__file__).resolve().parent / "text_features"


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": F.gelu,
    "relu": F.relu,
    "swish": swish,
}


class Attention(nn.Module):
    def __init__(self, config: ConfigDict, vis: bool = False):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer.get("num_heads", 12)
        hidden_size = config.get("hidden_size", 768)
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        attn_dropout = config.transformer.get("attention_dropout_rate", 0.0)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config: ConfigDict, input: int, hidden: int, output: int):
        super().__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, output)
        self.act_fn = ACT2FN["gelu"]
        dropout_rate = config.transformer.get("dropout_rate", 0.1) if hasattr(config, "transformer") else 0.1
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.act_fn(x)
        x = torch.sigmoid(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Convolutional ResNetV2 encoder wrapper preserving I-MMSeg module hierarchy."""

    def __init__(self, config: ConfigDict):
        super().__init__()
        self.hybrid = True
        self.config = config
        block_units = tuple(config.resnet.num_layers) if hasattr(config, "resnet") else (3, 4, 9)
        width_factor = float(config.resnet.width_factor) if hasattr(config, "resnet") else 1.0
        self.hybrid_model = ResNetV2(
            block_units=block_units,
            width_factor=width_factor,
            gradient_checkpointing=getattr(config, "gradient_checkpointing", False),
        )
        dropout_rate = config.transformer.get("dropout_rate", 0.1) if hasattr(config, "transformer") else 0.1
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, image: torch.Tensor):
        x, features = self.hybrid_model(image)
        x = self.dropout(x)
        return x, features


class Transformer(nn.Module):
    """I-MMSeg branch wrapper around Embeddings."""

    def __init__(self, config: ConfigDict, img_size: int = 128, vis: bool = False):
        super().__init__()
        self.embeddings = Embeddings(config)

    def forward(self, input_ids: torch.Tensor):
        return self.embeddings(input_ids)


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        use_batchnorm: bool = True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super().__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            head_channels * 2,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = tuple(config.decoder_channels)
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        n_skip = int(getattr(config, "n_skip", 3))
        if n_skip != 0:
            skip_channels = list(config.skip_channels)
            for i in range(4 - n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, enc_out: torch.Tensor, enc_features: Sequence[torch.Tensor] | None) -> torch.Tensor:
        x = self.conv_more(enc_out).contiguous()
        n_skip = int(getattr(self.config, "n_skip", 3))
        for i, decoder_block in enumerate(self.blocks):
            skip = enc_features[i] if (enc_features is not None and i < n_skip) else None
            x = decoder_block(x, skip=skip)
        return x


class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim: int, bias: bool = False):
        super().__init__()
        self.fusion_proj = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1, stride=1, bias=bias)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x_A: torch.Tensor, x_B: torch.Tensor, x_C: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_A, x_B, x_C], dim=1).contiguous()
        x = self.fusion_proj(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


def pixelwise_cosine_similarity(A: torch.Tensor, B: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Cosine similarity between corresponding pixels of two feature tensors (B, C, N)."""
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    dot_product = torch.matmul(A.transpose(1, 2), B)  # (B, N, N)
    norm_A = torch.norm(A, p=2, dim=1)  # (B, N)
    norm_B = torch.norm(B, p=2, dim=1)  # (B, N)
    denominator = torch.matmul(norm_A.unsqueeze(2), norm_B.unsqueeze(1))
    return dot_product / (denominator + epsilon)


def cosine_attention_update(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    attention_weights = torch.softmax(B, dim=-1)  # (B, N, N)
    A_transposed = A.transpose(1, 2)  # (B, N, C)
    aggregated = torch.matmul(attention_weights, A_transposed)  # (B, N, C)
    return aggregated.transpose(1, 2)  # (B, C, N)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * out


class CBAM(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_att(self.channel_att(x))


class CBAMFeatureReducer(nn.Module):
    def __init__(self, in_channels: int = 1024, reduced_channels: int = 64):
        super().__init__()
        self.cbam = CBAM(in_channels)
        self.conv1x1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cbam(x)
        x = self.conv1x1(x)
        x = self.global_avg_pool(x).view(x.size(0), -1)
        return x


class Class_Feature_Modulation(nn.Module):
    def __init__(self, text_dim: int = 32):
        super().__init__()
        self.fc = nn.Linear(text_dim, 8)

    def forward(self, x: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        params = self.fc(text_embed)
        gamma, beta = params.view(4, 2).chunk(2, dim=1)
        return x * (1 + gamma) + beta


class ForwardPredictionHead(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.config = config
        self.weight_nums = [4 * 4, 4 * 1]
        self.bias_nums = [4, 1]

        self.controller = nn.Conv2d(512, sum(self.weight_nums + self.bias_nums), kernel_size=1)
        self.Gap_64 = nn.Sequential(
            nn.GroupNorm(4, 64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 256, kernel_size=1),
        )
        self.GAP_256 = nn.Sequential(
            nn.GroupNorm(4, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 256, kernel_size=1),
        )
        self.GAP_512 = nn.Sequential(
            nn.GroupNorm(4, 512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 256, kernel_size=1),
        )
        self.precls_conv = nn.Sequential(
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, kernel_size=1),
        )

    def parse_dynamic_params(self, params: torch.Tensor, channels: int, weight_nums: list[int], bias_nums: list[int]):
        num_classes = params.size(0)
        num_layers = len(weight_nums)
        splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))
        weight_splits = splits[:num_layers]
        bias_splits = splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_classes * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_classes * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_classes, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_classes)
        return weight_splits, bias_splits

    def heads_forward(self, features: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor], num_classes: int) -> torch.Tensor:
        x = features
        n_layers = len(weights)
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_classes)
            if i < n_layers - 1:
                x = F.relu(x, inplace=True)
        return x

    def forward(self, feature_list: Sequence[torch.Tensor], clip_embedding: torch.Tensor, decoder_output: torch.Tensor) -> torch.Tensor:
        feat = (self.Gap_64(feature_list[2]) + self.GAP_256(feature_list[1]) + self.GAP_512(feature_list[0])) / 3.0
        batch_size = feat.shape[0]
        n_classes = int(getattr(self.config, "n_classes", 4))
        logits_array = []

        for i in range(batch_size):
            feat_embed = feat[i].unsqueeze(0).repeat(n_classes, 1, 1, 1)
            clip_embed = clip_embedding.unsqueeze(2).unsqueeze(2)
            vision_language_embedding = torch.cat([feat_embed, clip_embed], dim=1)
            params = self.controller(vision_language_embedding).squeeze(-1).squeeze(-1)

            decoder_out = decoder_output[i].unsqueeze(0)
            head_inputs = self.precls_conv(decoder_out).repeat(n_classes, 1, 1, 1)
            n_heads, _, h, w = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, h, w)

            weights, biases = self.parse_dynamic_params(params, 4, self.weight_nums, self.bias_nums)
            logits = self.heads_forward(head_inputs, weights, biases, n_heads)
            logits_array.append(logits.reshape(1, -1, h, w))

        return torch.cat(logits_array, dim=0)


def get_immseg_config() -> ConfigDict:
    """Default configuration for I-MMSeg."""
    config = ConfigDict()
    config.architecture = "immseg"
    config.model_name = "I-MMSeg"
    config.ablation = "IMMSEG"
    config.num_classes = 4
    config.n_classes = 4
    config.n_skip = 3
    config.hidden_size = 768
    config.vit_name = "R50-ViT-B_16"
    config.vit_patches_size = 16
    config.classifier = "seg"
    config.patches = ConfigDict({"size": (16, 16), "grid": (8, 8)})
    config.resnet = ConfigDict({"num_layers": (3, 4, 9), "width_factor": 1.0})
    config.transformer = ConfigDict({
        "num_heads": 12,
        "mlp_dim": 3072,
        "num_layers": 12,
        "attention_dropout_rate": 0.0,
        "dropout_rate": 0.1,
    })
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.gradient_checkpointing = False
    return config


class IMMSegNet(nn.Module):
    """I-MMSeg (Interaction Multi-Modal Segmentation) baseline architecture.

    Features:
    - 3-branch ResNetV2 hybrid encoder (CINE, PSIR, T2W)
    - Cross-modal cosine similarity attention update
    - Pre-computed CLIP vision-language class & modal embedding interaction
    - Dynamic convolution forward prediction head
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        img_size: int = 128,
        num_classes: int = 4,
        gradient_checkpointing: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        cfg = get_immseg_config()
        if config is not None:
            cfg.update(dict(config))
        cfg.n_classes = int(num_classes)
        cfg.num_classes = cfg.n_classes
        cfg.gradient_checkpointing = bool(gradient_checkpointing or cfg.get("gradient_checkpointing", False))
        cfg.patches.grid = (int(img_size / cfg.vit_patches_size), int(img_size / cfg.vit_patches_size))
        if "ablation" in kwargs:
            cfg.ablation = kwargs["ablation"]
        self.config = ConfigDict(cfg)
        self.num_classes = cfg.n_classes
        self.img_size = img_size

        # Load official text embeddings and register as buffers (automatic device/precision handling)
        class_emb_path = TEXT_FEATURES_DIR / "embedding_class_information.pth"
        modal_emb_path = TEXT_FEATURES_DIR / "embedding_MRI_information.pth"

        if class_emb_path.is_file():
            class_emb = torch.load(class_emb_path, map_location="cpu", weights_only=True).float()
        else:
            class_emb = torch.randn(4, 512)

        if modal_emb_path.is_file():
            modal_emb = torch.load(modal_emb_path, map_location="cpu", weights_only=True).float()
        else:
            modal_emb = torch.randn(3, 512)

        self.register_buffer("class_embedding", class_emb)
        self.register_buffer("modal_embedding", modal_emb)

        # Triple branch encoders
        self.transformer1 = Transformer(self.config, img_size)
        self.transformer2 = Transformer(self.config, img_size)
        self.transformer3 = Transformer(self.config, img_size)

        # Decoders
        self.decoder1 = DecoderCup(self.config)
        self.decoder2 = DecoderCup(self.config)
        self.decoder3 = DecoderCup(self.config)

        self.feature_fusion = nn.Sequential(
            OrderedDict([
                ("fusion1", Fusion_Embed(512)),
                ("fusion2", Fusion_Embed(256)),
                ("fusion3", Fusion_Embed(64)),
            ])
        )

        self.text_to_vision = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.modal_text_to_vision = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.modal_text_to_weight = nn.Sequential(
            Mlp(self.config, input=512, hidden=128, output=64),
            nn.ReLU(inplace=True),
        )

        self.GAP_cine = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.Flatten(start_dim=1),
        )
        self.GAP_psir = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.Flatten(start_dim=1),
        )
        self.GAP_t2w = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.Flatten(start_dim=1),
        )

        self.mix_cine = Fusion_Embed(1024)
        self.mix_psir = Fusion_Embed(1024)
        self.mix_t2w = Fusion_Embed(1024)

        self.CBAMFeatureReducer_cine = CBAMFeatureReducer()
        self.CBAMFeatureReducer_psir = CBAMFeatureReducer()
        self.CBAMFeatureReducer_t2w = CBAMFeatureReducer()
        self.Class_Feature_Modulation_cine = Class_Feature_Modulation()
        self.Class_Feature_Modulation_psir = Class_Feature_Modulation()
        self.Class_Feature_Modulation_t2w = Class_Feature_Modulation()

        self.class_MLP_dec_cine = Mlp(self.config, input=128, hidden=64, output=32)
        self.class_MLP_dec_psir = Mlp(self.config, input=128, hidden=64, output=32)
        self.class_MLP_dec_t2w = Mlp(self.config, input=128, hidden=64, output=32)

        self.class_MLP_cine = Mlp(self.config, input=512, hidden=256, output=64)
        self.class_MLP_psir = Mlp(self.config, input=512, hidden=256, output=64)
        self.class_MLP_t2w = Mlp(self.config, input=512, hidden=256, output=64)

        self.dec_output_fusion = Fusion_Embed(16)
        self.forward_prediction_head = ForwardPredictionHead(self.config)

    def _prepare_inputs(self, *inputs: torch.Tensor, **kwargs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(inputs) == 3:
            return inputs[0], inputs[1], inputs[2]
        if len(inputs) == 1:
            x = inputs[0]
            if x.shape[1] == 3:
                return x[:, 0:1], x[:, 1:2], x[:, 2:3]
            if x.shape[1] == 1 and "c0" in kwargs and "t2" in kwargs:
                return x, kwargs["c0"], kwargs["t2"]
        if "lge" in kwargs and "c0" in kwargs and "t2" in kwargs:
            # Note: I-MMSeg order is cine(c0), psir(lge), t2w(t2)
            return kwargs["c0"], kwargs["lge"], kwargs["t2"]
        raise ValueError(f"IMMSegNet expects 3 modalities (separate or 3-channel tensor), got {len(inputs)} inputs")

    def forward(self, *inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        c0, lge, t2 = self._prepare_inputs(*inputs, **kwargs)

        # ResNetV2 expects 3 input channels per modality
        cine = c0.repeat(1, 3, 1, 1) if c0.shape[1] == 1 else c0
        psir = lge.repeat(1, 3, 1, 1) if lge.shape[1] == 1 else lge
        t2w = t2.repeat(1, 3, 1, 1) if t2.shape[1] == 1 else t2

        modal_features = self.modal_text_to_vision(self.modal_embedding)
        modal_features_2 = self.modal_text_to_weight(self.modal_embedding)
        class_features = self.text_to_vision(self.class_embedding)

        # Encoders
        cine_f, cine_features = self.transformer1(cine)
        psir_f, psir_features = self.transformer2(psir)
        t2w_f, t2w_features = self.transformer3(t2w)

        batch_size, c, h, w = cine_f.shape
        modal_feature_cine = torch.cat([modal_features[0].repeat(batch_size, 1), self.GAP_cine(cine_f)], dim=1)
        modal_feature_psir = torch.cat([modal_features[1].repeat(batch_size, 1), self.GAP_psir(psir_f)], dim=1)
        modal_feature_t2w = torch.cat([modal_features[2].repeat(batch_size, 1), self.GAP_t2w(t2w_f)], dim=1)

        cine_mask = self.class_MLP_cine(modal_feature_cine).reshape(batch_size, h, w).unsqueeze(1).expand(-1, c, -1, -1)
        psir_mask = self.class_MLP_psir(modal_feature_psir).reshape(batch_size, h, w).unsqueeze(1).expand(-1, c, -1, -1)
        t2w_mask = self.class_MLP_t2w(modal_feature_t2w).reshape(batch_size, h, w).unsqueeze(1).expand(-1, c, -1, -1)

        cine_mask = torch.mul(cine_f, cine_mask).reshape(batch_size, c, -1)
        psir_mask = torch.mul(psir_f, psir_mask).reshape(batch_size, c, -1)
        t2w_mask = torch.mul(t2w_f, t2w_mask).reshape(batch_size, c, -1)

        cine_psir_cos_sim = torch.abs(pixelwise_cosine_similarity(cine_mask, psir_mask))
        cine_t2w_cos_sim = torch.abs(pixelwise_cosine_similarity(cine_mask, t2w_mask))
        psir_cine_cos_sim = torch.abs(pixelwise_cosine_similarity(psir_mask, cine_mask))
        psir_t2w_cos_sim = torch.abs(pixelwise_cosine_similarity(psir_mask, t2w_mask))
        t2w_cine_cos_sim = torch.abs(pixelwise_cosine_similarity(t2w_mask, cine_mask))
        t2w_psir_cos_sim = torch.abs(pixelwise_cosine_similarity(t2w_mask, psir_mask))

        cine_cp = cosine_attention_update(cine_mask, cine_psir_cos_sim).reshape(batch_size, c, h, w)
        cine_ct = cosine_attention_update(cine_mask, cine_t2w_cos_sim).reshape(batch_size, c, h, w)
        psir_pc = cosine_attention_update(psir_mask, psir_cine_cos_sim).reshape(batch_size, c, h, w)
        psir_pt = cosine_attention_update(psir_mask, psir_t2w_cos_sim).reshape(batch_size, c, h, w)
        t2w_tc = cosine_attention_update(t2w_mask, t2w_cine_cos_sim).reshape(batch_size, c, h, w)
        t2w_tp = cosine_attention_update(t2w_mask, t2w_psir_cos_sim).reshape(batch_size, c, h, w)

        cine_f = self.mix_cine(cine_f, cine_cp, cine_ct)
        psir_f = self.mix_psir(psir_f, psir_pc, psir_pt)
        t2w_f = self.mix_t2w(t2w_f, t2w_tc, t2w_tp)

        # Decoders
        dec_cine_out = self.decoder1(cine_f, cine_features)
        dec_psir_out = self.decoder2(psir_f, psir_features)
        dec_t2w_out = self.decoder3(t2w_f, t2w_features)

        # Modulating class_features across modalities
        cine_cbam = torch.mean(self.CBAMFeatureReducer_cine(cine_f), dim=0)
        psir_cbam = torch.mean(self.CBAMFeatureReducer_psir(psir_f), dim=0)
        t2w_cbam = torch.mean(self.CBAMFeatureReducer_t2w(t2w_f), dim=0)

        modal_dec_cine = torch.cat([modal_features_2[0], cine_cbam], dim=0)
        modal_dec_psir = torch.cat([modal_features_2[1], psir_cbam], dim=0)
        modal_dec_t2w = torch.cat([modal_features_2[2], t2w_cbam], dim=0)

        cine_text = self.class_MLP_dec_cine(modal_dec_cine)
        psir_text = self.class_MLP_dec_psir(modal_dec_psir)
        t2w_text = self.class_MLP_dec_t2w(modal_dec_t2w)

        class_features_cine = self.Class_Feature_Modulation_cine(class_features, cine_text)
        class_features_psir = self.Class_Feature_Modulation_psir(class_features, psir_text)
        class_features_t2w = self.Class_Feature_Modulation_t2w(class_features, t2w_text)

        img_features_fusion = [
            self.feature_fusion[i](cine_features[i], psir_features[i], t2w_features[i])
            for i in range(len(psir_features))
        ]

        out = self.dec_output_fusion(dec_cine_out, dec_psir_out, dec_t2w_out)
        out_seg = self.forward_prediction_head(img_features_fusion, class_features, out)

        return out_seg

    def forward_for_loss(self, *inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(*inputs, **kwargs)

    def load_pretrained_encoders(self, path: str | Path) -> None:
        """Load pretrained ResNet weights from R50-ViT-B_16.npz into all 3 branches."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Encoder checkpoint does not exist: {path}")
        with np.load(path, allow_pickle=False) as weights:
            encoders = [
                self.transformer1.embeddings.hybrid_model,
                self.transformer2.embeddings.hybrid_model,
                self.transformer3.embeddings.hybrid_model,
            ]
            encoders[0].validate_pretrained(weights)
            for encoder in encoders:
                encoder.load_from(weights)
            print(f"Loaded pretrained R50 weights from {path} into all 3 I-MMSeg branches.")
