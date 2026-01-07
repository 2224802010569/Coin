import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from feature.data.input.ccxt import CCXTInput
from feature.data.service.spark import SparkService
from feature.data.entities.candle import Candle
from feature.data.output.read import DataReadOutput
from feature.data.service.input import InputService
from feature.data.usecase.create import Create
from feature.data.usecase.update import Update
from feature.data.usecase.read import Read
from test_csv import csv_pd


if __name__ == "__main__":    
    df = CCXTInput().run()
    csv_pd(name="test", df=df)

    # Create().run()
    # Update().run()
    # df = Read().run()
    # to_csv(name = "read", df = df)

    # df = DataReadOutput().run()
    # to_csv(name = "data_read_output", df = df)
