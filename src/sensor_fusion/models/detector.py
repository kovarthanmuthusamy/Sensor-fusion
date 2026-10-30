"""The detector network: ResNet backbone, skip-connected upsampling neck, two heads (ADR 0009).

Photo (+ depth) in, two grids out, at 1/4 of the input size, in exactly the layout the target
encoder paints (`data/targets.py`, ADR 0006):
    class logits  `(B, num_classes, H/4, W/4)`  raw scores; apply the sigmoid for probabilities.
    regression    `(B, 5, H/4, W/4)`            channels in `REGRESSION_CHANNELS`, raw values.

The RGB baseline and the RGB+depth fusion model are this same class; only `in_channels` differs.
The extra depth channel starts with zero weights, so the fusion model begins as the baseline.
"""

import copy
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from sensor_fusion.data.targets import REGRESSION_CHANNELS

# ResNet-18 stage widths at strides 4, 8, 16, 32 (layer1..layer4).
_STAGE_CHANNELS = (64, 128, 256, 512)
_REQUIRED_MULTIPLE = 32  # the backbone halves the size five times
_HEAD_INIT_STD = 0.01  # last-layer weights start tiny, so the initial output is set by the bias


class Outputs(NamedTuple):
    class_logits: Tensor  # (B, num_classes, h, w)
    regression: Tensor  # (B, 5, h, w)


def _expand_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Copy a 3-channel first convolution to `in_channels`; channels beyond the third start at 0.

    Zero weights mean the added channel (depth) cannot change the output at the start, yet it still
    receives gradients, so the model learns how much to use it.
    """
    if in_channels < conv.in_channels:
        raise ValueError(f"cannot shrink the first convolution to {in_channels} channels")
    new = copy.deepcopy(conv)  # keeps kernel size, stride and padding without retyping them
    weight = torch.zeros(conv.out_channels, in_channels, *conv.kernel_size)
    weight[:, : conv.in_channels] = conv.weight.detach()
    new.in_channels = in_channels
    new.weight = nn.Parameter(weight)
    return new


def _conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _last_conv(head: nn.Sequential) -> nn.Conv2d:
    last = head[-1]
    assert isinstance(last, nn.Conv2d)
    return last


def _last_bias(head: nn.Sequential) -> Tensor:
    bias = _last_conv(head).bias
    assert bias is not None
    return bias


def _head(in_channels: int, hidden: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden, out_channels, 1),
    )


class Detector(nn.Module):
    """ResNet-18 -> skip-connected upsampling -> class head + regression head."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        neck_channels: int = 64,
        head_channels: int = 64,
        prior_prob: float = 0.1,
        distance_init_m: float = 30.0,
    ) -> None:
        """Args:
        in_channels: 3 = RGB, 4 = RGB + depth. num_classes: size of the class grid.
        backbone: only "resnet18" is supported. pretrained: load ImageNet weights (downloads
            once); tests and CI pass False.
        neck_channels / head_channels: widths of the upsampling path and of the two heads.
        prior_prob: initial probability of "object here" in every class cell. Starting at 0.5
            would let the millions of empty cells swamp the few object cells in the first steps.
        distance_init_m: the distance the regression head predicts at the start (the median
            distance of the labels, ADR 0006), so training starts near the right scale.
        """
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"unsupported backbone {backbone!r}; only 'resnet18' is built")
        if in_channels < 3:
            raise ValueError(f"in_channels must be 3 (RGB) or more, got {in_channels}")
        if not 0 < prior_prob < 1 or distance_init_m <= 0:
            raise ValueError("prior_prob must be in (0, 1) and distance_init_m positive")

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.conv1 = _expand_first_conv(net.conv1, in_channels)
        self.stem = nn.Sequential(net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])

        self.lateral = nn.ModuleList([nn.Conv2d(c, neck_channels, 1) for c in _STAGE_CHANNELS])
        self.smooth = nn.ModuleList(
            [_conv_bn_relu(neck_channels, neck_channels) for _ in _STAGE_CHANNELS[:-1]]
        )
        self.class_head = _head(neck_channels, head_channels, num_classes)
        self.regression_head = _head(neck_channels, head_channels, len(REGRESSION_CHANNELS))
        self._init_heads(prior_prob, distance_init_m)

    def _init_heads(self, prior_prob: float, distance_init_m: float) -> None:
        for head in (self.class_head, self.regression_head):
            last = _last_conv(head)
            nn.init.normal_(last.weight, std=_HEAD_INIT_STD)
            nn.init.zeros_(_last_bias(head))
        # sigmoid(bias) = prior_prob
        nn.init.constant_(_last_bias(self.class_head), -math.log((1 - prior_prob) / prior_prob))
        distance_channel = REGRESSION_CHANNELS.index("log_distance")
        with torch.no_grad():
            _last_bias(self.regression_head)[distance_channel] = math.log(distance_init_m)

    def forward(self, x: Tensor) -> Outputs:
        height, width = x.shape[-2:]
        if height % _REQUIRED_MULTIPLE or width % _REQUIRED_MULTIPLE:
            raise ValueError(
                f"input size {height}x{width} must be a multiple of {_REQUIRED_MULTIPLE}"
            )
        x = self.stem(self.conv1(x))
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        # Top-down path: upsample the coarse map, add the finer stage's detail, smooth.
        top = self.lateral[-1](features[-1])
        for level in range(len(features) - 2, -1, -1):
            top = F.interpolate(
                top, size=features[level].shape[-2:], mode="bilinear", align_corners=False
            )
            top = self.smooth[level](top + self.lateral[level](features[level]))
        return Outputs(self.class_head(top), self.regression_head(top))
