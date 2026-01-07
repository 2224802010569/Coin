import sys
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from config import PROFILE

class SignService:
    def __init__(self):
        self.horizon = PROFILE.Horizon

    def ma_sign(self, df: pd.DataFrame = None) -> pd.DataFrame:
        ma = df["close"].rolling(self.horizon, min_periods=1).mean()
        ma_slope = ma.diff()
        df["ma"] = ma
        df["ma_slope"] = ma_slope
        df["ma_direction"] = np.where(
            df["close"] > ma, "up",
            np.where(df["close"] < ma, "down", "neutral")
        )
        df["ma_slope_sign"] = np.where(ma_slope > 0, "up",np.where(ma_slope < 0, "down", "flat"))
        group_id = df["ma_direction"].ne(df["ma_direction"].shift()).cumsum()
        df["ma_persistence"] = df.groupby(group_id).cumcount() + 1
        return df[["ma_direction","ma_slope_sign","ma_persistence"]]


    def structure_sign(self, df: pd.DataFrame = None) -> pd.DataFrame:
        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)
        df["prev_high"] = prev_high
        df["prev_low"] = prev_low
        bullish = (df["high"] > prev_high) & (df["low"] > prev_low)
        bearish = (df["high"] < prev_high) & (df["low"] < prev_low)
        df["structure_state"] = np.select(
            [bullish, bearish],
            ["bullish", "bearish"],
            default="range"
        )
        df["failed_break"] = (
            ((df["high"] > prev_high) & (df["close"] < prev_high)) |
            ((df["low"] < prev_low) & (df["close"] > prev_low))
        )
        return df[["structure_state","failed_break"]]


    def vwap_sign(self, df: pd.DataFrame = None) -> pd.DataFrame:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        df["vwap"] = vwap
        df["vwap_distance"] = (df["close"] - vwap) / vwap
        df["vwap_location"] = np.where(
            df["vwap_distance"] > 0, "above",
            np.where(df["vwap_distance"] < 0, "below", "near")
        )
        df["vwap_stretch"] = pd.cut(
            df["vwap_distance"].abs(),
            bins=[0, 0.002, 0.005, 1],
            labels=["normal", "extended", "extreme"]
        )
        return df[["vwap_location","vwap_stretch"]]

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        ma = self.ma_sign(df)
        structure = self.structure_sign(df)
        vwap = self.vwap_sign(df)
        sign_df = pd.concat([ma, structure, vwap], axis=1)
        return sign_df