import random
import numpy as np
import torch

from feature.drl.agent.entities import Transition


def to_tensor(arr, device):
    return torch.tensor(arr, dtype=torch.float32, device=device)


def extract_action_mask(state, action_size: int):
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if arr.size < action_size:
        return np.ones(action_size, dtype=np.float32)
    mask = np.clip(arr[-action_size:], 0.0, 1.0)
    if float(mask.sum()) <= 0:
        return np.ones(action_size, dtype=np.float32)
    return mask.astype(np.float32)


def mask_q_values(q_values, valid_mask):
    q_arr = np.asarray(q_values, dtype=np.float32).copy()
    mask = np.asarray(valid_mask, dtype=np.float32).reshape(-1)
    invalid = mask <= 0
    q_arr[invalid] = -1e9
    return q_arr


def sample_batch(memory, batch_size: int):
    batch = random.sample(memory, batch_size)

    states = [item.state for item in batch]
    actions = [item.action for item in batch]
    rewards = [item.reward for item in batch]
    next_states = [item.next_state for item in batch]
    dones = [item.done for item in batch]

    return (
        np.stack(states),
        np.asarray(actions, dtype=np.int64),
        np.asarray(rewards, dtype=np.float32),
        np.stack(next_states),
        np.asarray(dones, dtype=np.float32),
    )


def clip_grad(model, max_norm: float = 1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
