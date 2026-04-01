from dataclasses import dataclass, field
from typing import Sequence

from config import ACTIVE_PROFILE, PROFILE_HIDDEN


def _hidden_sizes_for_profile(profile_name: str) -> Sequence[int]:
    match profile_name.lower():
        case "safe": sizes = PROFILE_HIDDEN.SAFE
        case "aggressive": sizes = PROFILE_HIDDEN.AGGRESSIVE
        case _: sizes = PROFILE_HIDDEN.BALANCED
    return tuple(int(v) for v in sizes[1:])


@dataclass(frozen=True)
class ModelConfig:
    hidden_sizes: Sequence[int] = field(
        default_factory=lambda: _hidden_sizes_for_profile(ACTIVE_PROFILE)
    )
    dropout: float = 0.2
    dueling: bool = True