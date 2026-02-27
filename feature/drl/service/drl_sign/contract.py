class ContractService:
    STATE_WINDOW_KEY = "window"
    STATE_PROFILE_KEY = "profile"
    WINDOW_FEATURES = [
        "close",
        "ma_low",
        "ma_high",
        "ema_low",
        "ema_high",
        "rsi",
        "volatility",
    ]
    PROFILE_FEATURES = [
        "Stability",
        "Volatility",
        "Aggression",
        "Horizon",
    ]
    ACTIONS = ["buy", "sell", "hold"]
    REWARD_KEYS = ["pnl", "drawdown"]
    METRICS = ["avg_pnl","winrate","max_drawdown","stability_score",]