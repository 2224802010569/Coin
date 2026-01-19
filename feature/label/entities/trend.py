from dataclasses import dataclass
from datetime import datetime

@dataclass
class Trend:
    timestamp: datetime
    timeframe: str
    label: str #uptrend / downtrend / sideways