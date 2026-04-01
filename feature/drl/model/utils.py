import torch
from torch import nn
from typing import Sequence
from feature.drl.model.entities import LayerSpec


def build_mlp(input_dim: int, hidden_sizes: Sequence[int], dropout: float):
    layers = []
    prev = input_dim

    for h in hidden_sizes:
        layers.extend([
            nn.Linear(prev, int(h)),
            nn.ReLU(),
        ])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = int(h)

    return nn.Sequential(*layers) if layers else nn.Sequential(nn.Identity())


def last_hidden_size(hidden_sizes: Sequence[int], input_dim: int) -> int:
    return int(hidden_sizes[-1]) if hidden_sizes else int(input_dim)


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)