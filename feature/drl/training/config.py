from dataclasses import dataclass
from pathlib import Path

from config import MODEL


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 10
    max_steps_per_episode: int | None = None
    model_path: Path = Path("model.pth")
    best_model_path: Path = Path("best_model.pth")
    train_timeframe: str = "1d"
    progress_bar_width: int = 28
    progress_update_every: int = 25
    resume_if_exists: bool = True
    cols = ["timestamp", "action", "action_name", "portfolio_value", "reward", "valid_actions"]
