from dataclasses import dataclass
from datetime import datetime

@dataclass
class Trend:
    start: datetime
    end: datetime
    timeframe: str
    label: str #uptrend / downtrend / sideways