"""Model B: ImageNet-pretrained ResNet-18, fully fine-tuned.

The final FC layer is replaced by a single-logit layer (sigmoid). For two classes this is
mathematically equivalent to a 2-way softmax, and it keeps every model's output format identical.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Detector(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)
        self.register_buffer("temperature", torch.ones(()))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x) / self.temperature
