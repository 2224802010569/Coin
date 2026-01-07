import pandas as pd
import numpy as np
from config import PROFILE
from feature.label.entities.strength import Strength
from feature.label.service.break_service import BreakService


class StrengthService:
    
    def __init__(self):
        self.horizon = PROFILE.Horizon
        self.confidence = PROFILE.Confidence

    def movement_score(self, df: pd.DataFrame) -> pd.Series:
        price_change = df["close"].diff(self.horizon)
        pct_change = price_change / df["close"].shift(self.horizon)
        score = pct_change.abs()
        return score

    def strong_or_weak(self, score: pd.Series) -> pd.Series:
        threshold = score.rolling(self.horizon).mean()
        return pd.Series(
            np.where(
                score > threshold * (1 + self.confidence),
                "strong",
                "weak"
            ),
            index=score.index
        )

    def high_or_low(
        self,strong_or_weak: pd.Series,
        break_df: pd.DataFrame
    ) -> pd.Series:
        labels = []
        for i in range(len(strong_or_weak)):
            base = strong_or_weak.iloc[i]
            quality = break_df.iloc[i]["break_quality"]
            if base == "weak":
                labels.append("weak")
                continue
            if quality in ["exhaustion"]:
                labels.append("strong high")
                continue
            if quality in ["failed", "weak"]:
                labels.append("strong low")
                continue
            labels.append("strong")
        return pd.Series(labels, index=strong_or_weak.index)

    def run(self,df: pd.DataFrame) -> pd.DataFrame:
        break_df = BreakService(df = df).run()
        score = self.movement_score(df)
        base = self.strong_or_weak(score)
        label_series = self.high_or_low(base, break_df)
        result = pd.DataFrame({"strength": label_series})
        segments = (result["strength"].ne(result["strength"].shift()).cumsum())
        output = []
        for seg_id, seg_df in result.groupby(segments):
            start_idx = seg_df.index[0]
            end_idx = seg_df.index[-1]
            output.append(
                Strength(
                    start=df.loc[start_idx, "timestamp"],
                    end=df.loc[end_idx, "timestamp"],
                    timeframe=df["timeframe"].iloc[0],
                    label=seg_df["strength"].iloc[0]
                )
            )
        return pd.DataFrame([s.__dict__ for s in output])
