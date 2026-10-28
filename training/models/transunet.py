"""TransUNet (Chen et al., 2021) Canonical Implementation for SCAR Benchmark.

Adapts the official VisionTransformer architecture from https://github.com/Beckschen/TransUNet
to the SCAR multi-modal 3-channel input contract and 4-class canonical segmentation.
"""
from __future__ import annotations

import copy
import logging
import math
from collections import OrderedDict
from os.path import join as pjoin
from typing import Any, Mapping

import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from ml_collections import ConfigDict

from training.models.backbones.resnet_v2 import ResNetV2

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Convert numpy weights to torch tensor, transposing HWIO to OIHW for conv kernels."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Attention(nn.Module):
    def __init__(self, config: ConfigDict, vis: bool = False):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
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
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct patch embeddings + position embeddings from ResNet features."""

    def __init__(self, config: ConfigDict, img_size: int = 128, in_channels: int = 3):
        super().__init__()
        self.config = config
        img_size_pair = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size_pair[0] // 16 // grid_size[0], img_size_pair[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size_pair[0] // patch_size_real[0]) * (img_size_pair[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size_pair[0] // patch_size[0]) * (img_size_pair[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor,
            )
            in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden, h, w)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config: ConfigDict, vis: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights: Mapping[str, Any], n_block: str) -> None:
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).view(-1)
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).view(-1)

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config: ConfigDict, vis: bool = False):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList([copy.deepcopy(Block(config, vis)) for _ in range(config.transformer["num_layers"])])
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis and weights is not None:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config: ConfigDict, img_size: int = 128, vis: bool = False):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor] | None]:
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class TransUNetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TransUNetDecoderCup(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, kernel_size=3, padding=1)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = list(decoder_channels)

        skip_channels = list(config.skip_channels)
        if self.config.n_skip != 0:
            for i in range(4 - self.config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        self.blocks = nn.ModuleList([
            TransUNetDecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ])

    def forward(self, hidden_states: torch.Tensor, features: list[torch.Tensor] | None = None) -> torch.Tensor:
        B, n_patch, hidden = hidden_states.size()
        h = w = int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < self.config.n_skip) else None
            x = decoder_block(x, skip=skip)
        return x


class TransUNetSegmentationHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        )


def get_transunet_config() -> ConfigDict:
    """Returns official R50-ViT-B_16 TransUNet configuration for 128x128 images."""
    config = ConfigDict()
    config.architecture = "transunet"
    config.model_name = "TransUNet"
    config.ablation = "TRANSUNET"
    config.num_classes = 4
    config.in_channels = 3
    config.img_size = 128
    config.vit_name = "R50-ViT-B_16"
    config.patch_size = 16
    config.patches = ConfigDict({"size": (16, 16), "grid": (8, 8)})
    config.hidden_size = 768

    config.transformer = ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.resnet = ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1.0

    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_skip = 3
    config.classifier = "seg"
    config.activation = "softmax"
    config.pretrained_path = None
    return config


class TransUNet2D(nn.Module):
    """Canonical TransUNet (R50-ViT-B_16) for SCAR multi-modal cardiac benchmark.

    Accepts 3 modalities (bSSFP, LGE, T2w) either as 1 concatenated tensor (B, 3, 128, 128)
    or 3 individual tensors (B, 1, 128, 128), and outputs 4-class logits (B, 4, 128, 128).
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        img_size: int = 128,
        num_classes: int = 4,
        zero_head: bool = False,
        vis: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        base_cfg = get_transunet_config()
        if config is not None:
            base_cfg.update(config)
        if "num_classes" in kwargs:
            num_classes = kwargs["num_classes"]
        base_cfg.num_classes = num_classes
        base_cfg.n_classes = num_classes
        base_cfg.img_size = img_size
        if "ablation" in kwargs:
            base_cfg.ablation = kwargs["ablation"]

        # Ensure grid is computed correctly for the specified img_size
        base_cfg.patches.grid = (img_size // 16, img_size // 16)
        self.config = base_cfg
        self.num_classes = num_classes
        self.img_size = img_size
        self.zero_head = zero_head

        self.transformer = Transformer(base_cfg, img_size=img_size, vis=vis)
        self.decoder = TransUNetDecoderCup(base_cfg)
        self.segmentation_head = TransUNetSegmentationHead(
            in_channels=base_cfg.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
        )

    def forward(self, *images: torch.Tensor) -> torch.Tensor:
        """Forward pass supporting either 1 tensor (B, 3, H, W) or 3 tensors (B, 1, H, W)."""
        if len(images) == 1:
            x = images[0]
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
        elif len(images) == 3:
            x = torch.cat(images, dim=1)
        else:
            raise ValueError(f"Expected 1 or 3 input tensors, got {len(images)}")

        x, attn_weights, features = self.transformer(x)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def forward_for_loss(self, *images: torch.Tensor) -> torch.Tensor:
        """Trainer hook: returns logits directly for loss calculation."""
        return self.forward(*images)

    def load_pretrained_encoders(self, pretrained_path: str | None = None) -> None:
        """Load pretrained Google ViT / ResNet weights from .npz checkpoint."""
        path = pretrained_path or self.config.get("pretrained_path")
        if not path:
            logger.info("TransUNet: No pretrained weights provided, using random He/Xavier initialization.")
            return

        logger.info("TransUNet: Loading pretrained weights from %s", path)
        weights = np.load(path)
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True)
            )
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"])
            )
            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"])
            )
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"])
            )

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                posemb_new.copy_(posemb)
            elif posemb.size(1) - 1 == posemb_new.size(1):
                posemb_new.copy_(posemb[:, 1:])
            else:
                logger.info("TransUNet: Resizing position embeddings from %s to %s", posemb.size(), posemb_new.size())
                ntok_new = posemb_new.size(1)
                _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1).numpy()
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb_new.copy_(torch.from_numpy(posemb_grid))

            # Load encoder blocks
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            # Load ResNetV2 hybrid backbone
            if self.transformer.embeddings.hybrid:
                res_weight = weights
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weight["conv_root/kernel"], conv=True)
                )
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

        logger.info("TransUNet: Successfully loaded pretrained weights!")
