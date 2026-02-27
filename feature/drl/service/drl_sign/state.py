import pandas as pd
from config import MODEL
from feature.data.entities.profile import PROFILE

class StateService:
    def __init__(self):
        self.window = MODEL.WINDOW

    def run(self, df, idx: int, profile):
        if idx < self.window:
            return None
        window_df = df.iloc[idx - self.window : idx]
        features = window_df[
            [
                "close",
                "ma_low",
                "ma_high",
                "ema_low",
                "ema_high",
                "rsi",
                "volatility",
            ]
        ].values
        profile_ctx = [
            profile.Stability,
            profile.Volatility,
            profile.Aggression,
            profile.Horizon,
        ]
        return {
            "window": features,
            "profile": profile_ctx,
        }
