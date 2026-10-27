"""ResNetV2 backbone with Weight Standardization and JAX/TransUNet weight loading."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from posixpath import join as pjoin
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def np2th(weights, conv=False):
    """Convert numpy weights to torch tensor, transposing HWIO to OIHW for conv kernels."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def _find_key(weights, key: str) -> str | None:
    """Resolve key across canonical JAX, Google ViT prefix, and Windows path separators."""
    candidates = [
        key,
        f"resnet/{key}",
        key.replace("/", "\\"),
        f"resnet\\{key.replace('/', chr(92))}",
    ]
    for cand in candidates:
        if cand in weights:
            return cand
    return None


class StdConv2d(nn.Conv2d):
    """Weight-standardized 2D convolution: W_std = (W - mean) / sqrt(var + 1e-5)."""

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """TransUNet hybrid bottleneck, retaining its historical class name.

    The source uses conv -> GroupNorm -> ReLU and a final residual ReLU,
    rather than a strict pre-activation ResNet-v2 block. Preserve this order
    for source checkpoint compatibility; the projection uses per-channel GN.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    @torch.no_grad()
    def load_from(self, weights, n_block, n_unit):
        prefix = pjoin(n_block, n_unit)

        def get_w(k, conv=False):
            actual_k = _find_key(weights, pjoin(prefix, k))
            if actual_k is None:
                raise ValueError(f"Pretrained encoder is missing {pjoin(prefix, k)}")
            return np2th(weights[actual_k], conv=conv)

        conv1_weight = get_w("conv1/kernel", conv=True)
        conv2_weight = get_w("conv2/kernel", conv=True)
        conv3_weight = get_w("conv3/kernel", conv=True)

        gn1_weight = get_w("gn1/scale")
        gn1_bias = get_w("gn1/bias")

        gn2_weight = get_w("gn2/scale")
        gn2_bias = get_w("gn2/bias")

        gn3_weight = get_w("gn3/scale")
        gn3_bias = get_w("gn3/bias")

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, "downsample"):
            proj_conv_weight = get_w("conv_proj/kernel", conv=True)
            proj_gn_weight = get_w("gn_proj/scale")
            proj_gn_bias = get_w("gn_proj/bias")

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """Three-stage, weight-standardized ResNetV2.

    Production units are (3, 4, 9), output stride 16 and width 1024. This
    truncated hybrid backbone is not a canonical four-stage ResNet-50.
    """

    def __init__(self, block_units, width_factor, gradient_checkpointing=False):
        super().__init__()
        width = int(64 * width_factor)
        if width <= 0 or width % 32:
            raise ValueError("ResNet width must be a positive multiple of 32 for GroupNorm")
        if len(block_units) != 3 or any(n < 1 or int(n) != n for n in block_units):
            raise ValueError("ResNet requires three positive integer block counts")
        self.width = width
        self.gradient_checkpointing = gradient_checkpointing
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.root = nn.Sequential(OrderedDict([
            ("conv", StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ("gn", nn.GroupNorm(32, width, eps=1e-6)),
            ("relu", nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ("block1", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f"unit{i:d}", PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width))
                 for i in range(2, block_units[0] + 1)],
            ))),
            ("block2", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f"unit{i:d}", PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2))
                 for i in range(2, block_units[1] + 1)],
            ))),
            ("block3", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f"unit{i:d}", PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4))
                 for i in range(2, block_units[2] + 1)],
            ))),
        ]))

    def _run_block(self, block, x):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, image):
        height, width = image.shape[2:]
        x = self.root(image)
        features = [x]
        x = self.pool(x)
        for i, block in enumerate(self.body):
            x = self._run_block(block, x)
            if i < 2:
                target_h, target_w = height // (4 * 2**i), width // (4 * 2**i)
                pad_h, pad_w = target_h - x.shape[2], target_w - x.shape[3]
                if not (0 <= pad_h < 3 and 0 <= pad_w < 3):
                    raise ValueError("Input dimensions are incompatible with the encoder skips")
                features.append(F.pad(x, (0, pad_w, 0, pad_h)))
        return x, features[::-1]

    def _pretrained_parameters(self):
        yield "conv_root/kernel", self.root.conv.weight, True
        yield "gn_root/scale", self.root.gn.weight, False
        yield "gn_root/bias", self.root.gn.bias, False
        for block_name, block in self.body.named_children():
            for unit_name, unit in block.named_children():
                prefix = f"{block_name}/{unit_name}"
                for i in (1, 2, 3):
                    yield f"{prefix}/conv{i}/kernel", getattr(unit, f"conv{i}").weight, True
                    yield f"{prefix}/gn{i}/scale", getattr(unit, f"gn{i}").weight, False
                    yield f"{prefix}/gn{i}/bias", getattr(unit, f"gn{i}").bias, False
                if hasattr(unit, "downsample"):
                    yield f"{prefix}/conv_proj/kernel", unit.downsample.weight, True
                    yield f"{prefix}/gn_proj/scale", unit.gn_proj.weight, False
                    yield f"{prefix}/gn_proj/bias", unit.gn_proj.bias, False

    def validate_pretrained(self, weights):
        """Check all required JAX keys/shapes before modifying the model."""
        for key, parameter, is_conv in self._pretrained_parameters():
            actual_key = _find_key(weights, key)
            if actual_key is None:
                raise ValueError(f"Pretrained encoder is missing {key}")
            tensor = np2th(weights[actual_key], conv=is_conv)
            if not is_conv:
                tensor = tensor.reshape(-1)
            if tensor.shape != parameter.shape:
                raise ValueError(f"Pretrained {key}: expected {tuple(parameter.shape)}, "
                                 f"got {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Pretrained encoder contains non-finite values: {key}")

    @torch.no_grad()
    def load_from(self, weights):
        self.validate_pretrained(weights)
        for key, parameter, is_conv in self._pretrained_parameters():
            actual_key = _find_key(weights, key)
            tensor = np2th(weights[actual_key], conv=is_conv)
            parameter.copy_(tensor if is_conv else tensor.reshape(-1))


# ==============================================================================
# Modality Normalization & FiLM Adapters for Shared-Weight Backbone
# ==============================================================================

MODALITY_ALIASES: dict[str | int, str] = {
    "cine": "cine",
    "bssfp": "cine",
    "c0": "cine",
    "0": "cine",
    0: "cine",
    "psir": "psir",
    "lge": "psir",
    "1": "psir",
    1: "psir",
    "t2w": "t2w",
    "t2": "t2w",
    "2": "t2w",
    2: "t2w",
}


def normalize_modality(modality: str | int) -> str:
    """Normalize modality names and indices to standard keys: 'cine', 'psir', 't2w'."""
    if isinstance(modality, str):
        key = modality.strip().lower()
    else:
        key = modality
    return MODALITY_ALIASES.get(key, str(modality).lower())


class FiLMAdapter(nn.Module):
    """Modality-specific Feature-wise Linear Modulation (FiLM) adapter.

    Applies affine channel scaling gamma_s^m * F + beta_s^m conditioned on modality m in {'cine', 'psir', 't2w'}.
    Initialized to identity (gamma=1.0, beta=0.0) so that representations are undisturbed before fine-tuning.
    """

    def __init__(self, channels: int, modalities: tuple[str, ...] = ("cine", "psir", "t2w")):
        super().__init__()
        self.channels = channels
        self.modalities = tuple(modalities)
        self.gamma = nn.ParameterDict({
            mod: nn.Parameter(torch.ones(1, channels, 1, 1)) for mod in modalities
        })
        self.beta = nn.ParameterDict({
            mod: nn.Parameter(torch.zeros(1, channels, 1, 1)) for mod in modalities
        })

    def forward(self, x: torch.Tensor, modality: str | int) -> torch.Tensor:
        mod = normalize_modality(modality)
        gamma = self.gamma.get(mod, None)
        beta = self.beta.get(mod, None)
        if gamma is None or beta is None:
            return x
        return gamma * x + beta


class SharedResNetV2(nn.Module):
    """Shared-Weight ResNetV2-50 backbone with independent 1-channel modality stems and FiLM adapters.

    Designed for multi-sequence cardiac magnetic resonance (CINE, PSIR/LGE, T2W):
    1. Modality Stems: 3 independent 1-channel stems (`stem_cine`, `stem_psir`, `stem_t2w`)
       consisting of StdConv2d(1, width, 7, stride=2, padding=3, bias=False) + GroupNorm + ReLU.
       Accommodates stark physical differences across CMR sequences (bSSFP steady-state vs PSIR
       inversion recovery vs T2 decay).
    2. Shared Body Trunk: 1 shared ResNetV2 trunk (block1, block2, block3) containing ~11.89M parameters,
       sharing 100% of convolutional kernels and GroupNorm layers across all modalities.
    3. Modality-Specific FiLM Adapters: Lightweight Feature-wise Linear Modulation (gamma * F + beta)
       at stages 1, 2, and 3, conditioned on modality ('cine', 'psir', 't2w'). Initialized to identity
       (gamma=1, beta=0) for zero perturbation at initialization.
    4. Memory & Footprint Efficiency: Slashes encoder parameter count from 35.68M (3 separate backbones)
       down to ~11.91M parameters (a 66.6% parameter cut, exceeding the 65.2% target).
    5. Direct JAX TransUNet Compatibility: Direct loading of ImageNet-21k pretrained weights into the shared
       trunk, with 1-channel stem kernels initialized via input channel averaging.
    """

    def __init__(
        self,
        config: Any = None,
        block_units: tuple[int, int, int] | None = None,
        width_factor: float | None = None,
        gradient_checkpointing: bool = False,
        modalities: tuple[str, ...] = ("cine", "psir", "t2w"),
    ):
        super().__init__()
        # Handle config vs positional arguments
        if config is not None:
            if isinstance(config, (tuple, list)) and len(config) == 3 and all(isinstance(x, int) for x in config):
                block_units = config
            elif hasattr(config, "resnet"):
                resnet_cfg = config.resnet
                if hasattr(resnet_cfg, "num_layers"):
                    block_units = resnet_cfg.num_layers
                elif isinstance(resnet_cfg, dict) and "num_layers" in resnet_cfg:
                    block_units = resnet_cfg["num_layers"]
                if hasattr(resnet_cfg, "width_factor"):
                    width_factor = resnet_cfg.width_factor
                elif isinstance(resnet_cfg, dict) and "width_factor" in resnet_cfg:
                    width_factor = resnet_cfg["width_factor"]
                gradient_checkpointing = getattr(config, "gradient_checkpointing", gradient_checkpointing)
            elif isinstance(config, dict) and "resnet" in config:
                resnet_cfg = config["resnet"]
                block_units = resnet_cfg.get("num_layers", block_units)
                width_factor = resnet_cfg.get("width_factor", width_factor)
                gradient_checkpointing = config.get("gradient_checkpointing", gradient_checkpointing)

        if block_units is None:
            block_units = (3, 4, 9)
        if width_factor is None:
            width_factor = 1.0

        width = int(64 * width_factor)
        if width <= 0 or width % 32:
            raise ValueError("ResNet width must be a positive multiple of 32 for GroupNorm")
        if len(block_units) != 3 or any(n < 1 or int(n) != n for n in block_units):
            raise ValueError("ResNet requires three positive integer block counts")

        self.width = width
        self.gradient_checkpointing = gradient_checkpointing
        self.modalities = tuple(modalities)

        # 1. Independent 1-channel stems
        num_gn_groups = min(32, width)
        self.stem_cine = nn.Sequential(OrderedDict([
            ("conv", StdConv2d(1, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ("gn", nn.GroupNorm(num_gn_groups, width, eps=1e-6)),
            ("relu", nn.ReLU(inplace=True)),
        ]))
        self.stem_psir = nn.Sequential(OrderedDict([
            ("conv", StdConv2d(1, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ("gn", nn.GroupNorm(num_gn_groups, width, eps=1e-6)),
            ("relu", nn.ReLU(inplace=True)),
        ]))
        self.stem_t2w = nn.Sequential(OrderedDict([
            ("conv", StdConv2d(1, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ("gn", nn.GroupNorm(num_gn_groups, width, eps=1e-6)),
            ("relu", nn.ReLU(inplace=True)),
        ]))
        self.stems = nn.ModuleDict({
            "cine": self.stem_cine,
            "psir": self.stem_psir,
            "t2w": self.stem_t2w,
        })

        # 2. Shared max-pooling and body trunk
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.body = nn.Sequential(OrderedDict([
            ("block1", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f"unit{i:d}", PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width))
                 for i in range(2, block_units[0] + 1)],
            ))),
            ("block2", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f"unit{i:d}", PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2))
                 for i in range(2, block_units[1] + 1)],
            ))),
            ("block3", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f"unit{i:d}", PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4))
                 for i in range(2, block_units[2] + 1)],
            ))),
        ]))

        # 3. Modality-specific FiLM adapters
        self.film1 = FiLMAdapter(width * 4, modalities=self.modalities)
        self.film2 = FiLMAdapter(width * 8, modalities=self.modalities)
        self.film3 = FiLMAdapter(width * 16, modalities=self.modalities)
        self.films = (self.film1, self.film2, self.film3)

    def _resolve_stem(self, modality_name: str | int) -> nn.Module:
        mod = normalize_modality(modality_name)
        if mod in self.stems:
            return self.stems[mod]
        stem_attr = getattr(self, f"stem_{mod}", None)
        if stem_attr is not None:
            return stem_attr
        return self.stem_cine

    def _run_block(self, block, x):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward_modality(
        self, x: torch.Tensor, modality_name: str | int = "cine"
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Process a single modality slice through its stem, the shared trunk, and FiLM adapters.

        Args:
            x: Input tensor of shape (B, 1, H, W).
            modality_name: Modality identifier ('cine', 'psir', 't2w' or aliases).

        Returns:
            bottleneck_feature: Shape (B, width * 16, H/16, W/16).
            skips: List of 3 skip connection features in descending order:
                   [skip0 (stage 2, B, width*8, H/8, W/8),
                    skip1 (stage 1, B, width*4, H/4, W/4),
                    skip2 (stem,    B, width,   H/2, W/2)]
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.shape[1] > 1:
            x = x[:, :1, :, :]

        height, width = x.shape[2:]
        stem = self._resolve_stem(modality_name)
        mod = normalize_modality(modality_name)

        # 1. Stem
        h = stem(x)
        features = [h]

        # 2. Pool + Shared Body + FiLM
        h = self.pool(h)
        for i, (block, film) in enumerate(zip(self.body, self.films)):
            h = self._run_block(block, h)
            h = film(h, mod)
            if i < 2:
                target_h, target_w = height // (4 * 2**i), width // (4 * 2**i)
                pad_h, pad_w = target_h - h.shape[2], target_w - h.shape[3]
                if not (0 <= pad_h < 3 and 0 <= pad_w < 3):
                    raise ValueError("Input dimensions are incompatible with the encoder skips")
                features.append(F.pad(h, (0, pad_w, 0, pad_h)))

        # features[::-1] = [block2 (skip0), block1 (skip1), stem (skip2)]
        return h, features[::-1]

    def forward(
        self, x: torch.Tensor, modality_name: str | int = "cine"
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self.forward_modality(x, modality_name)

    def forward_multimodal(
        self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor
    ) -> dict[str, tuple[torch.Tensor, list[torch.Tensor]]]:
        """Convenience method processing all 3 modalities in one call."""
        return {
            "cine": self.forward_modality(cine, "cine"),
            "psir": self.forward_modality(psir, "psir"),
            "t2w": self.forward_modality(t2w, "t2w"),
        }

    def _pretrained_body_parameters(self):
        for block_name, block in self.body.named_children():
            for unit_name, unit in block.named_children():
                prefix = f"{block_name}/{unit_name}"
                for i in (1, 2, 3):
                    yield f"{prefix}/conv{i}/kernel", getattr(unit, f"conv{i}").weight, True
                    yield f"{prefix}/gn{i}/scale", getattr(unit, f"gn{i}").weight, False
                    yield f"{prefix}/gn{i}/bias", getattr(unit, f"gn{i}").bias, False
                if hasattr(unit, "downsample"):
                    yield f"{prefix}/conv_proj/kernel", unit.downsample.weight, True
                    yield f"{prefix}/gn_proj/scale", unit.gn_proj.weight, False
                    yield f"{prefix}/gn_proj/bias", unit.gn_proj.bias, False

    def validate_pretrained(self, weights):
        """Check all required JAX trunk keys and shapes before modifying the model."""
        for key, parameter, is_conv in self._pretrained_body_parameters():
            actual_key = _find_key(weights, key)
            if actual_key is None:
                raise ValueError(f"Pretrained encoder is missing {key}")
            tensor = np2th(weights[actual_key], conv=is_conv)
            if not is_conv:
                tensor = tensor.reshape(-1)
            if tensor.shape != parameter.shape:
                raise ValueError(
                    f"Pretrained {key}: expected {tuple(parameter.shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Pretrained encoder contains non-finite values: {key}")

    @torch.no_grad()
    def load_from(self, weights):
        """Load pretrained JAX TransUNet weights into shared trunk and stems."""
        self.validate_pretrained(weights)
        for key, parameter, is_conv in self._pretrained_body_parameters():
            actual_key = _find_key(weights, key)
            tensor = np2th(weights[actual_key], conv=is_conv)
            parameter.copy_(tensor if is_conv else tensor.reshape(-1))

        # Initialize stems from conv_root and gn_root if present
        root_key = _find_key(weights, "conv_root/kernel")
        if root_key is not None:
            w_root = np2th(weights[root_key], conv=True)
            if w_root.ndim == 4 and w_root.shape[1] == 3:
                w_stem = w_root.mean(dim=1, keepdim=True)
                for stem in (self.stem_cine, self.stem_psir, self.stem_t2w):
                    if stem.conv.weight.shape == w_stem.shape:
                        stem.conv.weight.copy_(w_stem)

        gn_scale_key = _find_key(weights, "gn_root/scale")
        gn_bias_key = _find_key(weights, "gn_root/bias")
        if gn_scale_key is not None and gn_bias_key is not None:
            gn_scale = np2th(weights[gn_scale_key]).view(-1)
            gn_bias = np2th(weights[gn_bias_key]).view(-1)
            for stem in (self.stem_cine, self.stem_psir, self.stem_t2w):
                if stem.gn.weight.shape == gn_scale.shape:
                    stem.gn.weight.copy_(gn_scale)
                    stem.gn.bias.copy_(gn_bias)

    def load_from_transunet(self, weights_path_or_dict):
        """Load TransUNet JAX checkpoint from file path or loaded dictionary."""
        if isinstance(weights_path_or_dict, (str, Path)):
            path = Path(weights_path_or_dict)
            if not path.is_file():
                raise FileNotFoundError(f"Encoder checkpoint does not exist: {path}")
            with np.load(path, allow_pickle=False) as weights:
                self.load_from(weights)
        else:
            self.load_from(weights_path_or_dict)


__all__ = [
    "np2th",
    "StdConv2d",
    "conv3x3",
    "conv1x1",
    "PreActBottleneck",
    "ResNetV2",
    "FiLMAdapter",
    "SharedResNetV2",
    "normalize_modality",
    "MODALITY_ALIASES",
]
