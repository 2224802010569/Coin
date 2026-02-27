import pandas as pd
import torch
import torch.nn as nn
from config import PROFILE_HIDDEN
from feature.drl.service.drl_sign.contract import ContractService

class ModelService(nn.Module):
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
        self.head = nn.Linear(hidden[-1], len(ContractService.ACTIONS))
        self._build = True

    def forward(self, x, profile):
        if not self._build:
            self.build(profile)
        return self.head(self.backbone(x))
