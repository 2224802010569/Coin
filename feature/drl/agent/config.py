from dataclasses import dataclass

from config import MODEL


@dataclass(frozen=True)
class AgentConfig:
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    learning_rate: float = MODEL.LR
    batch_size: int = MODEL.BATCH_SIZE
    memory_size: int = 5000
    target_update_interval: int = 100
    min_replay_size: int = 128
    weight_decay: float = 1e-5
    use_double_dqn: bool = True
