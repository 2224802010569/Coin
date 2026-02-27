import pandas as pd

from feature.drl.input.profile_input import ProfileInput

class EvaluatorService:
    def __init__(self):
        self.profile = ProfileInput().run()

    def run(self, trade_metrics):
        if len(trade_metrics) == 0:
            return {}
        total_pnl = sum(m.pnl for m in trade_metrics)
        avg_pnl = total_pnl / len(trade_metrics)
        winrate = sum(m.is_win for m in trade_metrics) / len(trade_metrics)
        drawdowns = [m.pnl for m in trade_metrics if m.pnl < 0]
        max_drawdown = abs(min(drawdowns)) if drawdowns else 0.0
        stability_score = avg_pnl / (1.0 + max_drawdown)
        eval_key = self.profile.eval
        return {
            "avg_pnl": avg_pnl,
            "winrate": winrate,
            "max_drawdown": max_drawdown,
            "stability_score": stability_score,
            "eval_key": eval_key,
        }
