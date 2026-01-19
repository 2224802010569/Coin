import numpy as np

class EvalService:
    def run(self, rewards):
        rewards = np.array(rewards, dtype=float)
        equity = rewards.cumsum()
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        result = {
            "profit": equity[-1] if len(equity) > 0 else 0.0,
            "stability": np.mean(rewards) / (np.std(rewards) + 1e-8),
            "volatility": np.std(rewards),
            "max_drawdown": drawdown.min() if len(drawdown) > 0 else 0.0
        }
        return result
