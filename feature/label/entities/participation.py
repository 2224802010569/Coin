from dataclasses import dataclass
from datetime import datetime

@dataclass
class Participation:
    timestamp: datetime
    timeframe: str
    score: float
    label: str #strong / weak