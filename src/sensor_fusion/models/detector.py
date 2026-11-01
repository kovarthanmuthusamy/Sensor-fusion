"""The detector network: ResNet backbone, skip-connected upsampling neck, two heads (ADR 0009).

Photo (+ depth) in, two grids out, at 1/4 of the input size, in exactly the layout the target
encoder paints (`data/targets.py`, ADR 0006):
    class logits  `(B, num_classes, H/4, W/4)`  raw scores; apply the sigmoid for probabilities.
    regression    `(B, 5, H/4, W/4)`            channels in `REGRESSION_CHANNELS`, raw values.

The RGB baseline and the RGB+depth fusion model are this same class; only `in_channels` differs.
The extra input channels (depth, validity mask) go through their own first convolution
`conv1_extra`, whose output is added to the photo's. That equals one convolution over all
channels, but the depth filters become separate parameters with their own learning rate and
a zero start, so the fusion model begins as the baseline.
"""

import copy
import math
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from sensor_fusion.data.targets import REGRESSION_CHANNELS

# ResNet-18 stage widths at strides 4, 8, 16, 32 (layer1..layer4).
_STAGE_CHANNELS = (64, 128, 256, 512)
_REQUIRED_MULTIPLE = 32  # the backbone halves the size five times
# The output grid is 1/4 of the input (stem convolution x2, max-pool x2): the pooling block of the
# depth tower, and the `output_stride` of the preparation config.
_OUTPUT_STRIDE = 4
_HEAD_INIT_STD = 0.01  # last-layer weights start tiny, so the initial output is set by the bias


class Outputs(NamedTuple):
    class_logits: Tensor  # (B, num_classes, h, w)
    regression: Tensor  # (B, 5, h, w)


def _zero_conv_like(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """A convolution with the geometry of `conv` (kernel, stride, padding) but `in_channels`
    inputs, all weights zero.

    Zero weights mean the added channels (depth) cannot change the output at the start, yet they
    still receive gradients, so the model learns how much to use them.
    """
    new = copy.deepcopy(conv)  # keeps kernel size, stride and padding without retyping them
    new.in_channels = in_channels
    new.weight = nn.Parameter(torch.zeros(conv.out_channels, in_channels, *conv.kernel_size))
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


# A fill value larger than any scaled depth (which is in [0, 1]), used to ignore empty pixels when
# taking the nearest depth of a block.
_NO_DEPTH_FILL = 10.0


def pool_depth(depth: Tensor, mask: Tensor, block: int) -> Tensor:
    """Local LiDAR statistics on the output grid: `(B, 3, h, w)` with the mean depth of the valid
    pixels in each block, their nearest depth, and the fraction of pixels that are valid.

    `depth` and `mask` are `(B, 1, H, W)`; each output cell summarises a `block` x `block`
    block of pixels. `block` is a plain int so ONNX export sees a constant window.
    A block without any LiDAR value gets zeros everywhere, and the density channel says so.
    """
    kernel = (block, block)
    density = F.avg_pool2d(mask, kernel)
    mean = F.avg_pool2d(depth * mask, kernel) / density.clamp(min=1e-6)
    filled = depth + (1.0 - mask) * _NO_DEPTH_FILL
    nearest = torch.where(density > 0, -F.max_pool2d(-filled, kernel), torch.zeros_like(density))
    return torch.cat([mean, nearest, density], dim=1)


class _DepthTower(nn.Module):
    """Turns the pooled LiDAR statistics into features that are concatenated to the neck's output,
    so the heads see local depth directly instead of only through the early layers."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, depth: Tensor, mask: Tensor) -> Tensor:
        return self.body(pool_depth(depth, mask, _OUTPUT_STRIDE))


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
        depth_head: bool = False,
        depth_head_channels: int = 32,
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
        depth_head: also give both heads local LiDAR statistics through a small depth tower
            (needs depth in the input, `in_channels` >= 4). `depth_head_channels`: its width.
        """
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"unsupported backbone {backbone!r}; only 'resnet18' is built")
        if in_channels < 3:
            raise ValueError(f"in_channels must be 3 (RGB) or more, got {in_channels}")
        if not 0 < prior_prob < 1 or distance_init_m <= 0:
            raise ValueError("prior_prob must be in (0, 1) and distance_init_m positive")
        if depth_head and in_channels < 4:
            raise ValueError("depth_head needs a depth channel: in_channels must be 4 or 5")

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.in_channels = in_channels
        self.conv1 = net.conv1  # the photo, pretrained
        self.conv1_extra = _zero_conv_like(net.conv1, in_channels - 3) if in_channels > 3 else None
        self.stem = nn.Sequential(net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])

        self.lateral = nn.ModuleList([nn.Conv2d(c, neck_channels, 1) for c in _STAGE_CHANNELS])
        self.smooth = nn.ModuleList(
            [_conv_bn_relu(neck_channels, neck_channels) for _ in _STAGE_CHANNELS[:-1]]
        )
        tower_channels = depth_head_channels if depth_head else 0
        self.depth_tower = _DepthTower(depth_head_channels) if depth_head else None
        head_in = neck_channels + tower_channels
        self.class_head = _head(head_in, head_channels, num_classes)
        self.regression_head = _head(head_in, head_channels, len(REGRESSION_CHANNELS))
        self._init_heads(prior_prob, distance_init_m, neck_channels)

    def _init_heads(self, prior_prob: float, distance_init_m: float, neck_channels: int) -> None:
        for head in (self.class_head, self.regression_head):
            first = head[0]
            assert isinstance(first, nn.Conv2d)
            with torch.no_grad():  # depth-tower inputs start at zero: begin as without them
                first.weight[:, neck_channels:] = 0.0
            last = _last_conv(head)
            nn.init.normal_(last.weight, std=_HEAD_INIT_STD)
            nn.init.zeros_(_last_bias(head))
        # sigmoid(bias) = prior_prob
        nn.init.constant_(_last_bias(self.class_head), -math.log((1 - prior_prob) / prior_prob))
        distance_channel = REGRESSION_CHANNELS.index("log_distance")
        with torch.no_grad():
            _last_bias(self.regression_head)[distance_channel] = math.log(distance_init_m)

    def _load_from_state_dict(self, state_dict: dict[str, Tensor], prefix: str, *args: Any) -> None:
        """Accept checkpoints written when the depth channel was a 4th input of `conv1`: split
        that weight into the photo part and the extra part."""
        key = f"{prefix}conv1.weight"
        if key in state_dict and state_dict[key].shape[1] > 3:
            weight = state_dict[key]
            state_dict[key] = weight[:, :3]
            state_dict[f"{prefix}conv1_extra.weight"] = weight[:, 3:]
        super()._load_from_state_dict(state_dict, prefix, *args)

    def forward(self, x: Tensor) -> Outputs:
        height, width = x.shape[-2:]
        if height % _REQUIRED_MULTIPLE or width % _REQUIRED_MULTIPLE:
            raise ValueError(
                f"input size {height}x{width} must be a multiple of {_REQUIRED_MULTIPLE}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {x.shape[1]}")
        inputs = x
        first = self.conv1(x[:, :3])
        if self.conv1_extra is not None:
            first = first + self.conv1_extra(x[:, 3:])
        x = self.stem(first)
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
        if self.depth_tower is not None:
            depth = inputs[:, 3:4]
            mask = inputs[:, 4:5] if self.in_channels >= 5 else (depth > 0).to(depth.dtype)
            top = torch.cat([top, self.depth_tower(depth, mask)], dim=1)
        return Outputs(self.class_head(top), self.regression_head(top))
