import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from feature.deep_learning.service.preprocessing import PreprocessingService
from feature.label.entities.trend import Trend
from feature.label.entities.strength import Strength
from feature.deep_learning.input.data_input import DataInput
from test_csv import csv_pd


if __name__ == "__main__":

    df = DataInput().read()
    df_trend = DataInput().read(entity=Trend)
    p = PreprocessingService()
    # test = p.map_with_candle(candle= df, df = df_trend)
    # csv_pd(name="dl_tien_xu_ly", df=test)
    
    # df_str = DataInput().read(entity=Strength)
    # test = p.map_with_candle(candle= df, df = df_str)
    # csv_pd(name = "dl_tien_xu_ly", df=test)

    test = p.map_with_candle(candle= df, df = df_trend)
    test = p.sign_for_trend(df = test)
    csv_pd(name="dl_tien_xu_ly", df=test)