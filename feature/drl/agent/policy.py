import random
import numpy as np
import torch

class EpsilonGreedyPolicy:
    def __init__(self, action_size: int, epsilon: float = 1.0):
        self.action_size = int(action_size)
        self.epsilon = float(epsilon)

    def select_action(self, q_values, valid_mask=None, greedy=False) -> int:
        valid_mask = self._normalize_mask(valid_mask)
        valid_indices = np.flatnonzero(valid_mask > 0)
        if not greedy and random.random() < self.epsilon:
            return int(random.choice(valid_indices.tolist()))

        if isinstance(q_values, torch.Tensor):
            q_values = q_values.detach().cpu().numpy()

        q_arr = np.asarray(q_values, dtype=np.float32).copy()
        q_arr[valid_mask <= 0] = -1e9
        return int(np.argmax(q_arr))

    def update_epsilon(self, new_value: float):
        self.epsilon = float(new_value)

    def _normalize_mask(self, valid_mask):
        if valid_mask is None:
            return np.ones(self.action_size, dtype=np.float32)
        mask = np.asarray(valid_mask, dtype=np.float32).reshape(self.action_size,)
        mask = np.clip(mask, 0.0, 1.0)
        if float(mask.sum()) <= 0:
            return np.ones(self.action_size, dtype=np.float32)
        return mask
