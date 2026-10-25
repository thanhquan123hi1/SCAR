"""ResNetV2 backbone with Weight Standardization and JAX/TransUNet weight loading."""
from __future__ import annotations

from collections import OrderedDict
from posixpath import join as pjoin

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
