import pandas as pd
import numpy as np
from feature.label.service.sign import SignService


class BreakService:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.sign = SignService().run(df)

    def detect_break(self) -> pd.DataFrame:
        sign = self.sign
        break_direction = np.select(
            [
                (sign["structure_state"] == "bullish") & (~sign["failed_break"]),
                (sign["structure_state"] == "bearish") & (~sign["failed_break"]),
            ],
            ["up", "down"],
            default="none"
        )
        return pd.DataFrame({"break_direction": break_direction})

    def break_quality(self, break_df: pd.DataFrame) -> pd.DataFrame:
        sign = self.sign
        direction = break_df["break_direction"]
        break_quality = np.select(
            [
                sign["failed_break"],
                direction == "none",
                sign["vwap_stretch"] == "extreme",
                sign["vwap_stretch"] == "extended",
            ],
            [
                "failed",
                "none",
                "exhaustion",
                "strong",
            ],
            default="weak"
        )
        return pd.DataFrame({"break_quality": break_quality})

    def break_window(self, break_df: pd.DataFrame) -> pd.DataFrame:
        is_break = break_df["break_direction"] != "none"
        return pd.DataFrame({
            "is_break": is_break,
        })

    def run(self) -> pd.DataFrame:
        break_dir = self.detect_break()
        quality = self.break_quality(break_dir)
        window = self.break_window(break_dir)
        return pd.concat([break_dir, quality, window], axis=1)
