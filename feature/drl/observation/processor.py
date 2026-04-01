from __future__ import annotations
from dataclasses import replace
import numpy as np
import pandas as pd

from config import MODEL
from feature.drl.observation.config import ObservationConfig
from feature.drl.observation import utils


class ObservationProcessor:
    def __init__(self, config: ObservationConfig | None = None):
        window = getattr(MODEL, "WINDOW", 16)
        self.config = config or ObservationConfig(window_size=window)

    @property
    def feature_size(self) -> int:
        return len(self.config.columns)

    @property
    def flat_observation_size(self) -> int:
        # +3 cho mask (sell, hold, buy)
        base = self.config.window_size * self.feature_size + self.config.extra_state_size
        return base + 3

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        current_step: int,
        account_state=None,
        mask=None,
    ) -> np.ndarray:
        utils.validate_df(df, self.config.columns)
        step = int(np.clip(current_step, 0, len(df) - 1))
        wdf = utils.slice_windowwindow(df, step, self.config.window_size)
        vals = utils.extract_values(wdf, self.config.columns)
        vals = utils.normalize_block(vals, self.config.normalize, self.config.clip_value)
        vals = utils.pad_block(vals, self.config.window_size)
        extra = utils.build_extra_state(account_state, self.config.extra_state_size)
        mask_arr = self._normalize_mask(mask)
        return np.concatenate((vals.reshape(-1), extra, mask_arr))

    def _normalize_mask(self, mask):
        if mask is None:
            return np.array([1, 1, 1], dtype=np.float32)
        arr = np.asarray(mask, dtype=np.float32).reshape(3,)
        return np.clip(arr, 0, 1)

    def transform(self, observation) -> np.ndarray:
        arr = np.asarray(observation, dtype=np.float32).reshape(-1)
        return np.nan_to_num(arr,
            nan=0.0,
            posinf=self.config.clip_value,
            neginf=-self.config.clip_value,
        )

    def with_window(self, window_size: int) -> "ObservationProcessor":
        return ObservationProcessor(config=replace(self.config, window_size=window_size))

__all__ = ["ObservationProcessor", "ObservationConfig"]