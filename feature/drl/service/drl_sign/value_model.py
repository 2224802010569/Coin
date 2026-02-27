import pandas as pd
import torch
import torch.nn as nn
from config import PROFILE_HIDDEN

class ValueModelService(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = None
        self.head = None
        self._build = False

    def build(self, profile):
        profile_name = profile.name
        hidden = getattr(PROFILE_HIDDEN, profile_name.upper())
        layers = []
        for i in range(len(hidden) - 1):
            layers.append(nn.Linear(hidden[i], hidden[i + 1]))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden[-1], 1)
        self._build = True

    def forward(self, state, profile):
        if not self._build:
            self.build(profile)
        x_window = state["window"].reshape(state["window"].shape[0], -1)
        x_profile = torch.tensor(state["profile"], dtype=torch.float32).repeat(x_window.shape[0], 1)
        x = torch.cat([x_window, x_profile], dim=1)
        return self.head(self.backbone(x)).squeeze(-1)