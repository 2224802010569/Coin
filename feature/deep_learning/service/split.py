import pandas as pd
from typing import List, Tuple

class SplitService:
    def __init__(self, train_years=1, test_months=3, step_months=1):
        self.train_years = train_years
        self.test_months = test_months
        self.step_months = step_months

    def run(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        windows = []
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        start = df["timestamp"].min()
        end = df["timestamp"].max()
        current = start
        while True:
            train_end = current + pd.DateOffset(years=self.train_years)
            test_end = train_end + pd.DateOffset(months=self.test_months)
            if train_end >= end:
                break
            if test_end > end:
                # window cuối: chia 9:3
                remaining = df[df["timestamp"] >= current]
                if len(remaining) < 12:
                    break
                split_idx = int(len(remaining) * 0.75)
                train_df = remaining.iloc[:split_idx]
                test_df = remaining.iloc[split_idx:]
            else:
                train_df = df[
                    (df["timestamp"] >= current) &
                    (df["timestamp"] < train_end)
                ]
                test_df = df[
                    (df["timestamp"] >= train_end) &
                    (df["timestamp"] < test_end)
                ]
            if len(train_df) == 0 or len(test_df) == 0:
                break
            windows.append((train_df, test_df))
            current += pd.DateOffset(months=self.step_months)
        return windows
