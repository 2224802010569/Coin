from dataclasses import dataclass
from datetime import datetime

@dataclass
class Recommand:
    timestamp: datetime
    timeframe: str
    label: str          # buy / sell / hold
    vol: float          # %
    confidence: float   # độ tin cậy tín hiệu
