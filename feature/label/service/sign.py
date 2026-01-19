import numpy as np
import pandas as pd
from config import PROFILE

class SignService:
    def __init__(self):
        pass

    def map_with_candle(self,candle: pd.DataFrame = None,df: pd.DataFrame = None) -> pd.DataFrame:
        candle = candle.copy()
        df = df.copy()
        # candle["timestamp"] = pd.to_datetime(candle["timestamp"], utc=True)
        # df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        result = candle.merge(
            df[["timestamp", "label"]],
            on="timestamp",
            how="left"
        )
        return result


    def sign_for_trend(self, df: pd.DataFrame = None, k: int = 3, use_sideway: bool = True) -> pd.DataFrame:
        df["sign"] = "hold"
        highs = df["high"].values
        lows = df["low"].values
        for i in range(k, len(df) - k):
            trend = df.at[i, "label"]
            window_high = highs[i - k : i + k + 1]
            window_low = lows[i - k : i + k + 1]
            is_local_high = highs[i] == window_high.max()
            is_local_low = lows[i] == window_low.min()
            if trend == "uptrend" and is_local_high:
                df.at[i, "sign"] = "sell"
            elif trend == "downtrend" and is_local_low:
                df.at[i, "sign"] = "buy"
            elif trend == "sideways" and use_sideway:
                if is_local_low:
                    df.at[i, "sign"] = "buy"
                elif is_local_high:
                    df.at[i, "sign"] = "sell"
        return df

    def map_all(self,
        trend: pd.DataFrame = None,
        participation: pd.DataFrame = None
    ) -> pd.DataFrame:
        t = trend.copy()
        part = participation.copy()

        # ---- normalize column names ----
        for df in (t, part):
            df.columns = df.columns.str.strip().str.lower()

        # ---- ensure timestamp is column ----
        for df in (t, part):
            if "timestamp" not in df.columns and df.index.name == "timestamp":
                df.reset_index(inplace=True)

        result = t.merge(
            part[["timestamp", "timeframe", "score", "label"]]
                .rename(columns={
                    "score": "participation_score",
                    "label": "participation_label"
                }),
            on=["timestamp", "timeframe"],
            how="left"
        )

        return result