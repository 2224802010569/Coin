from dataclasses import dataclass, field
from typing import Sequence


DEFAULT_OBSERVATION_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ma_low",
    "ma_high",
    "ema_low",
    "ema_high",
    "rsi",
    "volatility",
)


@dataclass(frozen=True)
class EnvConfig:
    window_size: int = 16
    initial_balance: float = 100.0
    trading_fee: float = 0.001
    trade_fraction: float = 1.0
    min_cash_ratio: float = 0.10
    buy_target_exposure: float = 0.85
    sell_target_exposure: float = 0.15
    rebalance_tolerance: float = 0.05
    max_drawdown_penalty: float = 0.05
    invalid_action_penalty: float = 0.001
    hold_penalty: float = 0.0
    reward_scale: float = 1.0
    observation_columns: Sequence[str] = field(
        default_factory=lambda: DEFAULT_OBSERVATION_COLUMNS
    )
    price_column: str = "close"


ACTION_MAP = {
    0: "sell",
    1: "hold",
    2: "buy",
}
