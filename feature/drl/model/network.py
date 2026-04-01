from __future__ import annotations
from typing import Sequence

import torch
from torch import nn

from feature.drl.model.config import ModelConfig
from feature.drl.model import utils

class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        config: ModelConfig | None = None,
    ):
        super().__init__()
        self._validate_dims(input_dim, action_dim)

        self.config = config or ModelConfig()
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)

        self.feature_extractor = utils.build_mlp(
            input_dim=self.input_dim,
            hidden_sizes=self.config.hidden_sizes,
            dropout=self.config.dropout,
        )

        last_h = utils.last_hidden_size(self.config.hidden_sizes, self.input_dim)
        self._build_heads(last_h)

        self.apply(utils.init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0) if x.dim() == 1 else x
        z = self.feature_extractor(x.float())
        return self._forward_dueling(z) if self.config.dueling else self.q_head(z)

    @torch.no_grad()
    def act(self, x: torch.Tensor) -> torch.Tensor:
        q = self.forward(x)
        return torch.argmax(q, dim=1)

    def _build_heads(self, last_hidden: int):
        if not self.config.dueling:
            self.q_head = nn.Linear(last_hidden, self.action_dim)
            return

        self.value_head = nn.Sequential(
            nn.Linear(last_hidden, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(last_hidden, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, self.action_dim),
        )

    def _forward_dueling(self, features: torch.Tensor) -> torch.Tensor:
        value = self.value_head(features)
        adv = self.adv_head(features)
        return value + adv - adv.mean(dim=1, keepdim=True)

    @staticmethod
    def _validate_dims(input_dim: int, action_dim: int):
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")


__all__ = ["QNetwork", "ModelConfig"]