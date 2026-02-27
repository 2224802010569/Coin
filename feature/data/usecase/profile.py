import sqlite3
from config import DATA
from feature.data.entities.profile import PROFILE
from feature.data.usecase.read import Read
from feature.data.usecase.create import Create
from feature.data.usecase.update import Update
import pandas as pd

profiles = [
    {
        "name": "safe",
        "Stability": 0.75,
        "Volatility": 0.30,
        "Aggression": 0.25,
        "Confidence": 0.60,
        "Horizon": 90,
        "Min_Trend_length": 10,
        "eval": "stability",
    },
    {
        "name": "balanced",
        "Stability": 0.55,
        "Volatility": 0.50,
        "Aggression": 0.50,
        "Confidence": 0.55,
        "Horizon": 60,
        "Min_Trend_length": 5,
        "eval": "balanced",
    },
    {
        "name": "aggressive",
        "Stability": 0.30,
        "Volatility": 0.75,
        "Aggression": 0.75,
        "Confidence": 0.65,
        "Horizon": 30,
        "Min_Trend_length": 3,
        "eval": "profit",
    },
]
df = pd.DataFrame(profiles)

class ProfileUC:
    def __init__(self):
        pass

    def run(self) -> pd.DataFrame:
        with sqlite3.connect(DATA.STORAGE_DIR/ f"data.db") as conn:
            df = pd.read_sql(f"SELECT * FROM {'profile'}", conn)
            if df.empty or df is None:
                self.make()
                df = pd.read_sql(f"SELECT * FROM {'profile'}", conn)
        return df
    
    def make(self):
        Create().run(entity = PROFILE)
        Update().run(entity = PROFILE, db=df)
