import pandas as pd


class RewardService:
    def __init__(self):
        pass

    def run(self, step_info: dict, profile):
        pnl = step_info.get("pnl", 0.0)
        drawdown = step_info.get("drawdown", 0.0)
        stability_w = profile.Stability
        aggression_w = profile.Aggression
        return aggression_w * pnl - stability_w * abs(drawdown)
