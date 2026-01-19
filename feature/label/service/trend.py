import pandas as pd
import numpy as np
from feature.label.entities.trend import Trend
from config import PROFILE


class TrendService:
    def __init__(self):
        self.stability = PROFILE.Stability
        self.volatility = PROFILE.Volatility
        self.min_len = PROFILE.Min_Trend_length
        self.min_floor = 3
        self.vol_window = max(10, self.min_len)

    def _detect_raw_trends(self, df: pd.DataFrame):
        trends = []
        start, direction = None, None

        for i in range(1, len(df)):
            diff = df.loc[i, "close"] - df.loc[i - 1, "close"]
            cur = "up" if diff > 0 else "down" if diff < 0 else None
            if cur is None:
                continue

            if direction is None:
                start, direction = i - 1, cur
                continue

            if cur != direction:
                if i - start >= 2:
                    trends.append((start, i - 1, direction))
                start, direction = i - 1, cur

        if direction and start is not None and len(df) - start >= 2:
            trends.append((start, len(df) - 1, direction))

        return trends


    def _merge_trends(self, raws, df: pd.DataFrame):
        merged, i = [], 0
        while i < len(raws):
            s, e, d = raws[i]
            anchor = df.loc[s:e, "close"].median()
            violations = 0
            min_len = self.min_len
            max_gap = int(self.stability * self.min_len)
            j = i + 1
            while j < len(raws):
                ns, ne, _ = raws[j]
                if ns - e - 1 > max_gap:
                    break
                seg_median = df.loc[ns:ne, "close"].median()
                if (d == "up" and seg_median <= anchor) or \
                    (d == "down" and seg_median >= anchor):
                    break

                left = max(0, ns - self.vol_window)
                std = df.loc[left:ns, "close"].std()
                if std > 0 and abs(seg_median - anchor) > self.volatility * std:
                    violations += 1
                trend_len = ne - s + 1
                if violations > int((1 - self.stability) * trend_len):
                    min_len = max(self.min_floor, min_len - 1)
                e, anchor = ne, seg_median
                j += 1
            if e - s + 1 >= min_len:
                merged.append((s, e, d))
            i = j
        return merged

    def _detect_sideways(self, df: pd.DataFrame, trends):
        occupied = {i for s, e, _ in trends for i in range(s, e + 1)}
        sideways = []
        i = 0
        while i < len(df):
            if i in occupied:
                i += 1
                continue
            start, closes = i, []
            while i < len(df) and i not in occupied:
                closes.append(df.loc[i, "close"])
                i += 1
            if len(closes) >= self.min_floor:
                mid = np.median(closes)
                if mid > 0 and (max(closes) - min(closes)) / mid <= self.volatility:
                    sideways.append((start, i - 1))
        return sideways

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        timeframe = df["timeframe"].iloc[0]
        trends = self._merge_trends(self._detect_raw_trends(df), df)
        sideways = self._detect_sideways(df, trends)
        labels = ["sideways"] * len(df)
        for s, e, d in trends:
            label = "uptrend" if d == "up" else "downtrend"
            for i in range(s, e + 1):
                labels[i] = label
        for s, e in sideways:
            for i in range(s, e + 1):
                labels[i] = "sideways"
        result = [
            Trend(
                timestamp=df.loc[i, "timestamp"],
                timeframe=timeframe,
                label=labels[i]
            )
            for i in range(len(df))
        ]
        return pd.DataFrame([r.__dict__ for r in result])
