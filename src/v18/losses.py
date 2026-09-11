from __future__ import annotations

import torch
from torch import Tensor, nn


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt((prediction - target).square() + self.epsilon**2).mean()


def build_loss(config: dict) -> nn.Module:
    if config.get("name", "").lower() != "charbonnier":
        raise ValueError(f"Unsupported loss: {config.get('name')!r}")
    return CharbonnierLoss(float(config["epsilon"]))

