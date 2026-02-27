import pandas as pd
import torch
from config import MODEL, DATA
from feature.drl.service.drl_sign.episode import EpisodeService
from feature.drl.service.drl_sign.model import ModelService
from feature.drl.service.drl_sign.ppo import PPOService
from feature.drl.service.drl_sign.reward import RewardService
from feature.drl.service.drl_sign.state import StateService
from feature.drl.service.drl_sign.value_model import ValueModelService
from feature.drl.service.re_learing.evaluator import EvaluatorService


class TrainerService:
    def __init__(self):
        self.state_service = StateService()
        self.reward_service = RewardService()
        self.evaluator = EvaluatorService()
        self.model_path = DATA.MODEL_DIR

    def run(self, df, profile):
        policy = ModelService()
        value = ValueModelService()
        policy.build(profile)
        value.build(profile)
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters()),
            lr=MODEL.LR
        )
        ppo = PPOService()
        buffer = EpisodeService()
        horizon = int(profile.Horizon)
        for _ in range(MODEL.EPOCHS):
            episode = {"steps": []}
            for idx in range(len(df)):
                state = self.state_service.run(df, idx, profile)
                if state is None:
                    continue
                logits = policy(state, profile)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
                step_info = df.iloc[idx].to_dict()
                reward = self.reward_service.run(step_info, profile)
                value_est = value.run(state, profile)
                episode["steps"].append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "logp": logp.detach(),
                    "value": value_est.detach()
                })
                if len(episode["steps"]) >= horizon:
                    metrics = self.evaluator.run(step_info.get("metrics", []), profile)
                    episode["metrics"] = metrics
                    buffer.add_episode(episode)
                    episode = {"steps": []}
            top_eps = buffer.get_all()[:MODEL.REPLAY_TOP_K]
            for ep in top_eps:
                batch = self._build_batch(ep["steps"])
                ppo.run(batch, policy, value, optimizer, profile)
        model_path = self.model_path / profile.name / f"model.pt2"
        torch.save(policy.state_dict(), model_path)

    def _build_batch(self, steps):
        return {
            "state": [s["state"] for s in steps],
            "action": torch.stack([s["action"] for s in steps]),
            "reward": torch.tensor([s["reward"] for s in steps]),
            "logp": torch.stack([s["logp"] for s in steps]),
        }
