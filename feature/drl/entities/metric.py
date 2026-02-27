from dataclasses import dataclass

@dataclass
class TradeMetric:
    label: str
    vol: float
    pnl: float
    is_win: bool
