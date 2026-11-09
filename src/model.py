"""CatDogNet: a small CNN whose parameters are programmed by hand.

Autograd is disabled globally: no gradients, no optimizers, no backprop.
Label convention: 0 = cat, 1 = dog; sigmoid(logit) is P(dog).
"""

from __future__ import annotations

import torch
from torch import nn

torch.set_grad_enabled(False)


DEFAULT_WIDTHS = (16, 32, 32)   # wsp-001..003; wsp-004 uses a wider variant (documented design change)


class CatDogNet(nn.Module):
    def __init__(self, widths: tuple[int, int, int] = DEFAULT_WIDTHS) -> None:
        super().__init__()
        c1, c2, c3 = widths
        self.widths = tuple(widths)
        self.conv1 = nn.Conv2d(3, c1, 5, padding=2)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(4)
        self.head = nn.Linear(c3 * 16, 1)
        for param in self.parameters():
            param.requires_grad_(False)
            param.zero_()  # discard framework random init; everything is programmed

    def forward_with_activations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        acts: dict[str, torch.Tensor] = {"input": x}
        acts["conv1"] = self.conv1(x)
        acts["relu1"] = torch.relu(acts["conv1"])
        acts["pool1"] = self.pool(acts["relu1"])
        acts["conv2"] = self.conv2(acts["pool1"])
        acts["relu2"] = torch.relu(acts["conv2"])
        acts["pool2"] = self.pool(acts["relu2"])
        acts["conv3"] = self.conv3(acts["pool2"])
        acts["relu3"] = torch.relu(acts["conv3"])
        acts["gap"] = self.gap(acts["relu3"]).flatten(1)
        acts["logit"] = self.head(acts["gap"]).squeeze(1)
        acts["prob_dog"] = torch.sigmoid(acts["logit"])
        return acts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_activations(x)["logit"]
