from profile import Profile
import sqlite3
import sys
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from config import DATA
from feature.data.input.ccxt import CCXTInput
from feature.data.entities.candle import Candle
from feature.data.output.read import DataReadOutput
from feature.data.service.input import InputService
from feature.data.usecase.create import Create
from feature.data.usecase.update import Update
from feature.data.usecase.read import Read

from feature.label.entities.trend import Trend
from test_csv import csv_pd


if __name__ == "__main__":    
    # df = CCXTInput().run()
    # csv_pd(name="test", df=df)

    # Create().run()
    # Update().run()
    # df = Read().run()
    # csv_pd(name = "data", df = df)
    
    with sqlite3.connect(DATA.STORAGE_DIR/ f"data.db") as conn:
        df = pd.read_sql(f"SELECT * FROM {'profile'}", conn)
    csv_pd(name = "data_profile", df = df)
