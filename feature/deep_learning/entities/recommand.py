from dataclasses import dataclass
from datetime import datetime


@dataclass
class Sign:
    timestamp: datetime
    timeframe: str
    vol: str #large / small
    label: str #buy / sell / hold

@dataclass
class Recommand:
    timestamp: datetime
    timeframe: str
    vol: str #large / small
    label: str #buy / sell / hold