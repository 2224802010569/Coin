import pandas as pd
import torch
from feature.data.entities.profile import PROFILE
from feature.drl.entities.recommand import Recommand
from feature.drl.service.drl_sign.inference import InferenceService
from feature.drl.service.drl_sign.state import StateService
from feature.drl.service.drl_vol.vol import VolService
from feature.drl.service.drl_sign.contract import ContractService

class ExecutionService:
    def __init__(self):
        self.state_service = StateService()
        self.signal_inf = InferenceService()
        self.vol = VolService()
        self.metrics = []

    def load_signal_model(self, profile: PROFILE):
        self.signal_inf.load(profile)

    def run(self, df, idx, profile) -> Recommand:
        state = self.state_service.run(df, idx, profile)
        if state is None:
            return None
        logits = self.signal_inf.run(state, profile)
        dist = torch.distributions.Categorical(logits=logits)
        probs = torch.softmax(logits, dim=-1)
        mean_prob = probs.mean(dim=0)
        label_idx = torch.argmax(mean_prob).item()
        confidence = mean_prob[label_idx].item()
        label = ContractService.ACTIONS[label_idx]
        raw_state = df.iloc[idx].to_dict()
        vol = self.vol.predict(
            state=raw_state,
            signal=label,
            position=raw_state.get("position", 0.0),
        )
        return Recommand(
            timestamp=raw_state["timestamp"],
            timeframe=raw_state["timeframe"],
            label=label,
            vol=vol,
            confidence=confidence,
        )
