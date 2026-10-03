"""Custom CNN feature extractor for the (4, 8, 8) Block Blast observation.

Stable-Baselines3's default NatureCNN expects at least ~36x36 inputs, so a
small board-specific CNN is used instead. It processes the board spatially
(three 3x3 convolutions with padding, preserving the 8x8 resolution) and
produces a 256-dim feature vector for the policy/value heads.
"""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class BlockBlastCNN(BaseFeaturesExtractor):
    """Conv(4->64) -> Conv(64->64) -> Conv(64->128) -> FC(->features_dim)."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = int(self.cnn(sample).shape[1])

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
