"""Tiny ConvNeXt-style encoder with TensorRT-friendly Conv/BN/ReLU/Add ops.

Four outputs have strides 4, 8, 16 and 32. The default channels/depths
are (20, 40, 80, 160)/(1, 1, 3, 1). Combined with enet_dec/global restore,
the full model has 982,012 PyTorch parameters; static 512x1024 ONNX reports
0.9703M parameters and 11.6798 GFLOPs (strict organizer gate PASSED).
See docs/student_backbones.md for all three variants.
"""
from __future__ import annotations

import torch
from torch import nn


class _MicroBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, 4 * channels, 1, bias=False),
            nn.BatchNorm2d(4 * channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ConvNeXtMicro(nn.Module):
    channels = (20, 40, 80, 160)
    depths = (1, 1, 3, 1)

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, self.channels[0], 4, stride=4, bias=False),
                                  nn.BatchNorm2d(self.channels[0]))
        self.downsamples = nn.ModuleList(
            nn.Sequential(nn.Conv2d(a, b, 2, stride=2, bias=False), nn.BatchNorm2d(b))
            for a, b in zip(self.channels[:-1], self.channels[1:]))
        self.stages = nn.ModuleList(
            nn.Sequential(*(_MicroBlock(c) for _ in range(n)))
            for c, n in zip(self.channels, self.depths))

    def forward(self, x: torch.Tensor):
        outputs = []
        x = self.stem(x)
        for index, stage in enumerate(self.stages):
            if index:
                x = self.downsamples[index - 1](x)
            x = stage(x)
            outputs.append(x)
        return tuple(outputs)
