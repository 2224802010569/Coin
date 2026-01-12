import pandas as pd
from typing import Optional

class PreprocessingService:
    def __init__(self):
        pass

    def map_with_candle(self, candle: pd.DataFrame = None, df: pd.DataFrame = None) -> pd.DataFrame:
        candle = candle.copy()
        candle["label"] = None
        idx = 0
        for idx_c, c in candle.iterrows():
            while idx < len(df):
                i = df.iloc[idx]
                if i["start"] <= c["timestamp"] <= i["end"]:
                    candle.at[idx_c, "label"] = i["label"]
                    break
                if c["timestamp"] > i["end"]:
                    idx += 1
                    candle.at[idx_c, "label"] = None
                else:
                    break
        return candle

    def sign_for_trend(self, df: pd.DataFrame = None, k: int = 3, use_sideway: bool = True) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
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

            