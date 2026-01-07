import pandas as pd
import numpy as np
from config import PROFILE
from feature.label.entities.trend import Trend


class TrendService:

    def __init__(self):
        self.stability = PROFILE.Stability      # càng cao → càng ít nhiễu
        self.volatility = PROFILE.Volatility    # scale nhiễu
        self.min_len = PROFILE.Min_Trend_length

    # --------------------------------------------------
    # TRUE RANGE (tối giản)
    # --------------------------------------------------
    def _true_range(self, h, l, pc):
        return max(h - l, abs(h - pc), abs(l - pc))

    # --------------------------------------------------
    # HƯỚNG GIÁ (close-to-close, có eps)
    # --------------------------------------------------
    def _price_dir(self, prev_close, close, eps):
        if close > prev_close + eps:
            return 1
        if close < prev_close - eps:
            return -1
        return 0

    # --------------------------------------------------
    # RUN
    # --------------------------------------------------
    def run(self, df: pd.DataFrame, timeframe: str = None) -> pd.DataFrame:
        df = df.reset_index(drop=True)

        segments = []

        state = "sideways"
        state_start = 0

        # ---------- pha accumulate ----------
        cand_dir = 0
        cand_len = 0

        # ---------- pha trend ----------
        trend_dir = 0
        trend_len = 0
        noise_count = 0
        tr_list = []

        prev_close = df.loc[0, "close"]

        for i in range(1, len(df)):
            row = df.loc[i]
            close = row["close"]

            # ----- ATR cục bộ -----
            tr = self._true_range(row["high"], row["low"], prev_close)
            tr_list.append(tr)
            atr = np.mean(tr_list)
            eps = self.volatility * atr if atr > 0 else 0.0

            d = self._price_dir(prev_close, close, eps)

            # ==================================================
            # PHASE 1: ACCUMULATE (SIDEWAYS)
            # ==================================================
            if state == "sideways":
                if d == 0:
                    cand_len = 0
                    cand_dir = 0
                else:
                    if cand_dir == 0 or d == cand_dir:
                        cand_dir = d
                        cand_len += 1
                    else:
                        cand_dir = d
                        cand_len = 1

                # ---- xác nhận trend ----
                if cand_len >= self.min_len:
                    state = "uptrend" if cand_dir == 1 else "downtrend"
                    state_start = i - cand_len + 1

                    # khóa trend
                    trend_dir = cand_dir
                    trend_len = cand_len
                    noise_count = 0
                    tr_list = []

            # ==================================================
            # PHASE 2: TREND (DUY TRÌ)
            # ==================================================
            else:
                trend_len += 1

                if d == 0:
                    noise_count += 1
                elif d != trend_dir:
                    noise_count += 2   # nến ngược hướng → phạt nặng

                max_noise = int(trend_len * (1 - self.stability))

                # ---- FAIL ----
                if noise_count > max_noise:
                    segments.append(
                        Trend(
                            start=df.loc[state_start, "timestamp"],
                            end=df.loc[i - 1, "timestamp"],
                            timeframe=timeframe,
                            label=state
                        )
                    )

                    # reset toàn bộ
                    state = "sideways"
                    state_start = i
                    cand_dir = 0
                    cand_len = 0
                    trend_dir = 0
                    trend_len = 0
                    noise_count = 0
                    tr_list = []

            prev_close = close

        # đóng đoạn cuối
        segments.append(
            Trend(
                start=df.loc[state_start, "timestamp"],
                end=df.loc[len(df) - 1, "timestamp"],
                timeframe=timeframe,
                label=state
            )
        )

        return pd.DataFrame([t.__dict__ for t in segments])
