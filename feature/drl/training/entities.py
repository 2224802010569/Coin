from dataclasses import dataclass

@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    mean_loss: float
    epsilon: float


@dataclass(frozen=True)
class TrainingState:
    completed_episodes: int = 0
    best_reward: float = float("-inf")
