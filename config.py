from datetime import datetime
from pathlib import Path

class DATA:
    BASE_DIR = Path(__file__).resolve().parents[0]
    STORAGE_DIR = BASE_DIR/ "data" / "storage"
    INPUT_DIR = BASE_DIR / "data" / "storage" / "input"
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
    END_DATE= datetime(2024, 1, 1)

class PROFILE:
    Stability        = 0.55 #Ổn định
    Volatility       = 0.50 #Biến động
    Aggression       = 0.50 #Hung hãng
    Confidence       = 0.55 #Lạc quan và bi quan
    Horizon          = 60
    Min_Trend_length = 14

class LABEL:
    TREND= ["uptrend" , "downtrend" , "sideways"]
    STRENGTH= ["strong" , "weak" , "strong high" , "weak high" , "strong low" , "weak low"]
    RECOMMEND = ["buy", "sell"]

class SIGN:
    pass

DATA = DATA()
COIN = COIN()
PROFILE = PROFILE()
LABEL = LABEL()