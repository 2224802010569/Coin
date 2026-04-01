from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from feature.drl.agent.config import AgentConfig
from feature.drl.agent.policy import EpsilonGreedyPolicy
from feature.drl.model.network import QNetwork
from feature.drl.agent.entities import Transition
from feature.drl.agent.utils import (
    to_tensor,
    sample_batch,
    clip_grad,
    extract_action_mask,
)


class DQNAgent:
    def __init__(self, state_size: int, action_size: int, config=None, device=None):
        self.config = config or AgentConfig()
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._init_networks()
        self._init_training_utilities()

    def _init_networks(self):
        self.policy_net = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def _init_training_utilities(self):
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = deque(maxlen=self.config.memory_size)
        self.policy = EpsilonGreedyPolicy(self.action_size, self.config.epsilon_start)
        self.train_steps = 0

    @property
    def epsilon(self) -> float:
        return self.policy.epsilon

    def select_action(self, state, greedy: bool = False) -> int:
        state_t = to_tensor(state, self.device).unsqueeze(0)
        valid_mask = extract_action_mask(state, self.action_size)
        with torch.no_grad():
            q = self.policy_net(state_t).squeeze(0)
        return self.policy.select_action(q_values=q, valid_mask=valid_mask, greedy=greedy)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    def train_step(self) -> float | None:
        if len(self.memory) < max(self.config.min_replay_size, self.config.batch_size):
            return None

        loss = self._train_batch()
        self._update_target_network()
        self._decay_epsilon()
        return loss

    def save(self, path, metadata: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint_state(metadata=metadata), path)
        return path

    def load(self, path, strict=True):
        ck = torch.load(Path(path), map_location=self.device)
        self.policy_net.load_state_dict(ck["policy_state_dict"], strict=strict)
        self.target_net.load_state_dict(ck.get("target_state_dict", ck["policy_state_dict"]), strict=strict)
        if ck.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        self.policy.update_epsilon(float(ck.get("epsilon", self.config.epsilon_end)))
        self.train_steps = int(ck.get("train_steps", 0))
        return ck

    def _train_batch(self) -> float:
        states, actions, rewards, next_states, dones = sample_batch(
            self.memory, self.config.batch_size
        )
        s = to_tensor(states, self.device)
        a = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        r = to_tensor(rewards, self.device)
        ns = to_tensor(next_states, self.device)
        d = to_tensor(dones, self.device)
        q = self.policy_net(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            next_masks = torch.clamp(ns[:, -self.action_size:], 0.0, 1.0)
            if self.config.use_double_dqn:
                next_policy_q = self.policy_net(ns)
                next_policy_q = next_policy_q.masked_fill(next_masks <= 0, -1e9)
                next_actions = next_policy_q.argmax(dim=1, keepdim=True)
                next_target_q = self.target_net(ns).gather(1, next_actions).squeeze(1)
                nq = next_target_q
            else:
                next_target_q = self.target_net(ns).masked_fill(next_masks <= 0, -1e9)
                nq = next_target_q.max(dim=1).values
            tgt = r + (1 - d) * self.config.gamma * nq
        loss = self.loss_fn(q, tgt)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad(self.policy_net)
        self.optimizer.step()
        self.train_steps += 1
        return float(loss.item())

    def _update_target_network(self):
        if self.train_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _decay_epsilon(self):
        new_eps = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        self.policy.update_epsilon(new_eps)

    def _checkpoint_state(self, metadata: dict | None = None):
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "train_steps": self.train_steps,
        }
        if metadata:
            checkpoint["metadata"] = metadata
        return checkpoint


__all__ = ["DQNAgent", "AgentConfig"]
