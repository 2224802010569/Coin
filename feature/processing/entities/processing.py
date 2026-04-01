from dataclasses import dataclass
from datetime import datetime

@dataclass
class Processing:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str
    return_1d: float
    volatility: float
    ma_low: float
    ma_high: float
    ema_low: float
    ema_high: float
    rsi: float
