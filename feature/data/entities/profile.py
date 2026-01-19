from dataclasses import dataclass


@dataclass
class PROFILE:
    name: str
    Stability: float #Ổn định
    Volatility: float #Biến động
    Aggression: float #Hung hãng
    Confidence: float #Lạc quan và bi quan
    Horizon: int
    Min_Trend_length: int
    eval: str