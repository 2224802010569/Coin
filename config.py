from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

class DATA:
    BASE_DIR = Path(__file__).resolve().parents[0]
    STORAGE_DIR = BASE_DIR/ "data" / "storage"
    INPUT_DIR = BASE_DIR / "data" / "storage" / "input"
    MODEL_DIR = BASE_DIR / "data" / "storage" / "model"
    PGSQL = {
        "engine": "postgres",
        "host": "localhost",
        "port": 5432,
        "database": "db",
        "user": "user",
        "password": "password",
    },
    INPUT = "ccxt"
    OUTPUT = "sql"

class COIN:
    TIMEFRAME = ["1d"]
    SYMBOL = "BTC/USDT"
    EXCHANGE = "binance"
    START_DATE= datetime(2018, 1, 1)
    END_DATE= datetime(2025, 1, 1)
    BALANCE = 100
    FEE = 0.001

@dataclass(frozen=True)
class PROFILE:
    Stability: float #Ổn định
    Volatility: float #Biến động
    Aggression: float #Hung hãng
    Confidence: float #Lạc quan và bi quan
    Horizon: int
    Min_Trend_length: int
    eval: str

SAFE = PROFILE(
    Stability=0.75,
    Volatility=0.30,
    Aggression=0.25,
    Confidence=0.60,
    Horizon=90,
    Min_Trend_length=10,
    eval = "stability"
)

BALANCED = PROFILE(
    Stability=0.55,
    Volatility=0.50,
    Aggression=0.50,
    Confidence=0.55,
    Horizon=60,
    Min_Trend_length=5,
    eval = "balanced"
)

AGGRESSIVE = PROFILE(
    Stability=0.30,
    Volatility=0.75,
    Aggression=0.75,
    Confidence=0.65,
    Horizon=30,
    Min_Trend_length=3,
    eval = "profit"
)

class PROCESSING:
    MA = [7,21]
    EMA = [7,21]
    RSI = 14
    VOLATILITY = 14
    TRAIN = 365
    TEST = 90
    STEP = 90

class LABEL:
    TREND= ["uptrend" , "downtrend" , "sideways"]
    STRENGTH= ["strong" , "weak" , "strong high" , "weak high" , "strong low" , "weak low"]
    RECOMMEND = ["buy", "sell"]

class MODEL:
    EPOCHS = 20
    BATCH_SIZE = 256
    LR = 1e-3
    TRAIN_RATIO = 0.7
    WINDOW = 16
    EPISODE_MEMORY = 50
    REPLAY_TOP_K = 5

class PROFILE_HIDDEN:
    SAFE = [11, 64, 32, 16]
    BALANCED = [11, 64, 64, 32]
    AGGRESSIVE = [11, 128, 64, 32]


class EXECUTION:
    DEFAULT_VOL = 0.1
    HOLD_VOL = 0.0

def get_profile(name:str = "balanced") -> PROFILE:
    name = name.lower()
    return {
        "safe": SAFE,
        "balanced": BALANCED,
        "aggressive": AGGRESSIVE
    }[name]

def set_profile(name:str = "balanced"):
    global ACTIVE_PROFILE, PROFILE
    ACTIVE_PROFILE = name
    PROFILE = get_profile(name)

DATA = DATA()
COIN = COIN()
ACTIVE_PROFILE = "balanced"
PROFILE = get_profile(ACTIVE_PROFILE)
PROCESSING = PROCESSING()
LABEL = LABEL()
MODEL = MODEL()