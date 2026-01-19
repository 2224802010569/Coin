from dataclasses import dataclass
from datetime import datetime


@dataclass
class Recommand:
    timestamp: datetime
    timeframe: str
    vol: float #%
    label: str #buy / sell / hold