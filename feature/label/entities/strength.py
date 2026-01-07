from dataclasses import dataclass
from datetime import datetime

@dataclass
class Strength:
    start: datetime
    end: datetime
    timeframe: str
    label: str #strong / weak / strong high / weak high / strong low / weak low