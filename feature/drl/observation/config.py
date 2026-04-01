from dataclasses import dataclass, field
from typing import Sequence

from feature.drl.env.config import DEFAULT_OBSERVATION_COLUMNS


@dataclass(frozen=True)
class ObservationConfig:
    window_size: int = 16
    columns: Sequence[str] = field(default_factory=lambda: DEFAULT_OBSERVATION_COLUMNS)
    extra_state_size: int = 6
    clip_value: float = 10.0
    normalize: bool = True
