import os
import plotly.graph_objects as go
import pandas as pd

from feature.data.output.read import DataReadOutput

class TestDraw:
    
    def __init__(self):
        self.df = DataReadOutput().run()

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

        df = trend_df.sort_values("timestamp").reset_index(drop=True)

        start_idx = 0
        current_label = df.loc[0, "label"]

        for i in range(1, len(df)):
            if df.loc[i, "label"] != current_label:
                fig.add_vrect(
                    x0=df.loc[start_idx, "timestamp"],
                    x1=df.loc[i - 1, "timestamp"],
                    fillcolor=color_map.get(current_label, "rgba(0,0,0,0)"),
                    opacity=1,
                    layer="below",
                    line_width=0
                )
                start_idx = i
                current_label = df.loc[i, "label"]

        # đoạn cuối
        fig.add_vrect(
            x0=df.loc[start_idx, "timestamp"],
            x1=df.loc[len(df) - 1, "timestamp"],
            fillcolor=color_map.get(current_label, "rgba(0,0,0,0)"),
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

    def draw_with_sign(self, df: pd.DataFrame) -> go.Figure:
        fig = self._base_candlestick()
        sign_df = df[df["sign"].isin(["buy", "sell"])]
        color_map = {
            "buy": "green",
            "sell": "red",
        }
        symbol_map = {
            "buy": "triangle-up",
            "sell": "triangle-down",
        }
        y_map = {
            "buy": "low",
            "sell": "high",
        }
        for sign, gdf in sign_df.groupby("sign"):
            fig.add_trace(
                go.Scatter(
                    x=gdf["timestamp"],
                    y=gdf[y_map[sign]],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=color_map[sign],
                        symbol=symbol_map[sign]
                    ),
                    name=f"sign:{sign}"
                )
            )
        fig.update_layout(
            title="Price with Buy / Sell Signals"
        )
        return fig
