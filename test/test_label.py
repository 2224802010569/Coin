import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from feature.data.entities.candle import Candle
from feature.label.entities.trend import Trend
from feature.label.input.data_input import DataInput
from feature.label.service.participation import ParticipationService
from feature.label.service.trend import TrendService
from feature.label.usecase.label_uc import Label
from feature.label.output.label import LabelOutput
from feature.label.service.sign import SignService
from test_csv import csv_pd
from test_draw import TestDraw


if __name__ == "__main__":
    df = DataInput().read(entity = Candle)

    # df = ParticipationService().run(df = df)
    # csv_pd(name="label_participation", df=df)

    tre_df = TrendService().run(df = df)
    # vi = TestDraw(df)
    # fig1 = vi.draw_with_trend(tre_df)
    # fig1.write_html("test_ans/trend.html")

    # str_df = StrengthService().run(df= df)
    # vi = TestDraw(df)
    # fig2 = vi.draw_with_strength(str_df)
    # fig2.write_html("test_ans/strength.html")
    
    # df = LabelOutput().read_trend()
    # csv_pd(name="label_trend", df=df)

    # df = LabelOutput().read_participation()
    # csv_pd(name="label_strength", df=df)

    # df = LabelOutput().read_all()
    df = Label().all()
    csv_pd(name="label_all", df=df)

    # df = SignService().map_with_candle(candle = df, df = tre_df)
    # csv_pd(name = "label_map_candle_trend", df = df)

    # df = SignService().map_with_candle(candle = df, df = tre_df)
    # df = SignService().sign_for_trend(df = df)
    # csv_pd(name="sign_for_trend", df=df)
    # fig = TestDraw().draw_with_sign(df)
    # fig.write_html("test_ans/sign_trend.html")