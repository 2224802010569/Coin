import pandas as pd
import numpy as np
from feature.label.entities.trend import Trend
from config import PROCESSING


class ProcessingService:
    def __init__(self):
        self.ma = PROCESSING.MA
        self.ema = PROCESSING.EMA
        self.vol = PROCESSING.VOLATILITY
        self.rsi = PROCESSING.RSI

    def run(self, df: pd.DataFrame = None) -> pd.DataFrame:
        ma_low, ma_high = self.run_ma(df=df)
        df["ma_low"] = ma_low
        df["ma_high"] = ma_high
        ema_low, ema_high = self.run_ema(df=df)
        df["ema_low"] = ema_low
        df["ema_high"] = ema_high
        df["rsi"] = self.run_rsi(df=df)
        df["volatility"] = self.run_vol(df=df)
        return df
    
    def run_ma(self, df: pd.DataFrame = None):
        windows = self.ma
        w_low, w_high = int(windows[0]), int(windows[1])
        ma_low = df["close"].rolling(window=w_low, min_periods=1).mean()
        ma_high = df["close"].rolling(window=w_high, min_periods=1).mean()
        return ma_low, ma_high

    def run_ema(self, df: pd.DataFrame = None):
        windows = list(self.ema)
        w_low, w_high = int(windows[0]), int(windows[1])
        ema_low = df["close"].ewm(span=w_low, adjust=False).mean()
        ema_high = df["close"].ewm(span=w_high, adjust=False).mean()
        return ema_low, ema_high

    def run_rsi(self, df: pd.DataFrame = None):
        period = self.rsi
        delta = df["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_gain = avg_gain.combine_first(gain.ewm(alpha=1/period, adjust=False).mean())
        avg_loss = loss.combine_first(loss.ewm(alpha=1/period, adjust=False).mean())
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(0)
        return rsi

    def run_vol(self, df: pd.DataFrame = None):
        window = self.vol
        log_return = np.log(df["close"] / df["close"].shift(1))
        volatility = log_return.rolling(window=window, min_periods=1).std() * np.sqrt(window)
        volatility = volatility.fillna(0)
        return volatility