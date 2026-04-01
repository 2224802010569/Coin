from dataclasses import dataclass

@dataclass(frozen=True)
class TradeResult:
    action_name: str
    traded_cash: float = 0.0
    traded_units: float = 0.0
    fee_paid: float = 0.0
    invalid: bool = False