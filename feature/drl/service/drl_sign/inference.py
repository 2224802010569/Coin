import os
import numpy as np
import pandas as pd
import torch
from config import DATA
from feature.drl.service.drl_sign.model import ModelService

class InferenceService:
    def __init__(self):
        self.policy = ModelService()

    def load(self, profile):
        policy = self.policy
        policy.build(profile)
        model_path = DATA.MODEL_DIR / profile.name / "policy.pth"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location="cpu")
            self.policy.load_state_dict(state_dict)
        else:
            torch.save(policy.state_dict(), model_path)
        self.policy.eval()

    def run(self, state, profile):
        x_window = torch.tensor(state["window"], dtype=torch.float32)
        x_profile = torch.tensor(
            np.asarray(state["profile"]), dtype=torch.float32
        ).view(1, -1).repeat(x_window.shape[0], 1)
        x = torch.cat(
            [x_window.view(x_window.shape[0], -1), x_profile],
            dim=1
        )
        with torch.no_grad():
            logits = self.policy(x, profile)
            return logits



