import pandas as pd
import numpy as np
from config import PROFILE
from feature.label.entities.participation import Participation


class ParticipationService:
    def __init__(self):
        self.window = PROFILE.Min_Trend_length
        self.percentile = self.calculate_percentile()

    def calculate_percentile(self) -> float: 
        h = PROFILE.Horizon
        c = PROFILE.Confidence
        a = PROFILE.Aggression
        v = PROFILE.Volatility
        s = PROFILE.Stability
        risk = (0.3*a + 0.25*v + 0.25*c + 0.2*(1-s))
        adj = max(0.0, min((h-30)/120,1.0))
        base = 75
        per = base - 15*risk + 5*adj
        return max(55.0, min(per, 85.0))

    def run(self, df: pd.DataFrame, timeframe: str = "1d") -> pd.DataFrame:
        df = df.copy()
        df["spread"] = (df["close"] - df["open"]).abs()
        df["vol_ma"] = df["volume"].rolling(self.window).mean()
        df["spread_ma"] = df["spread"].rolling(self.window).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma"]
        df["spread_ratio"] = df["spread"] / df["spread_ma"]
        df["score"] = round(df["vol_ratio"] * df["spread_ratio"],5)
        df["score"] = df["score"].replace([np.inf, -np.inf], np.nan)
        participations = []
        for i in range(len(df)):
            if i < self.window:
                continue
            window_scores = df.loc[i - self.window : i - 1, "score"].dropna()
            if len(window_scores) < self.window:
                continue
            threshold = np.percentile(window_scores, self.percentile)
            score = df.at[i, "score"]
            label = "strong" if score >= threshold else "weak"
            participations.append(
                Participation(
                    timestamp = df.at[i, "timestamp"],
                    timeframe = timeframe,
                    score = float(score),
                    label = label
                )
            )
        return pd.DataFrame([p.__dict__ for p in participations])
