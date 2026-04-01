from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ObservationParts:
    window: np.ndarray
    state: np.ndarray