import os
import plotly.graph_objects as go
import pandas as pd

class TestDraw:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _base_candlestick(self) -> go.Figure:
        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=self.df["timestamp"],
                open=self.df["open"],
                high=self.df["high"],
                low=self.df["low"],
                close=self.df["close"],
                name="Price"
            )
        )

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=20)
        )

        return fig

    def draw_with_trend(self, trend_df: pd.DataFrame) -> go.Figure:
        fig = self._base_candlestick()

        color_map = {
            "uptrend": "rgba(0, 200, 0, 0.15)",
            "downtrend": "rgba(200, 0, 0, 0.15)",
            "sideways": "rgba(150, 150, 150, 0.10)"
        }

        for _, row in trend_df.iterrows():
            fig.add_vrect(
                x0=row["start"],
                x1=row["end"],
                fillcolor=color_map.get(row["label"], "rgba(0,0,0,0)"),
                opacity=1,
                layer="below",
                line_width=0
            )

        fig.update_layout(
            title="Price with Trend State (Background)"
        )
        return fig

    def draw_with_strength(self, strength_df: pd.DataFrame) -> go.Figure:
        fig = self._base_candlestick()

        color_map = {
            "strong": "green",
            "weak": "gray",
            "strong high": "darkred",
            "strong low": "orange",
            "weak high": "lightcoral",
            "weak low": "gold"
        }

        symbol_map = {
            "strong": "triangle-up",
            "weak": "circle",
            "strong high": "triangle-up",
            "strong low": "triangle-up",
            "weak high": "circle",
            "weak low": "circle"
        }

        for _, row in strength_df.iterrows():
            mask = self.df["timestamp"] == row["start"]
            if not mask.any():
                continue
            price = self.df.loc[mask, "close"].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[row["start"]],
                    y=[price],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=color_map.get(row["label"], "black"),
                        symbol=symbol_map.get(row["label"], "circle")
                    ),
                    name=row["label"]
                )
            )
        fig.update_layout(
            title="Price with Strength Markers"
        )
        return fig
