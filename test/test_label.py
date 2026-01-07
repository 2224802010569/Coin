import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from feature.label.input.data_input import DataInput
from feature.label.service.sign import SignService
from feature.label.service.break_service import BreakService
from feature.label.service.strength import StrengthService
from feature.label.service.trend import TrendService
from test_csv import csv_pd
from test_draw import TestDraw


if __name__ == "__main__":
    df = DataInput().read()
    # df = SignService().ma_sign(df)
    # csv_pd(name="label_ma_sign", df=df)
    # df = SignService().structure_sign(df)
    # csv_pd(name="label_structure_sign", df=df)
    # df = SignService().vwap_sign(df)
    # csv_pd(name="label_vwap_sign", df=df)
    # df = BreakService(df).run()
    # df = TrendService().run(df = df)
    # csv_pd(name="label_trend", df=df)

    tre_df = TrendService().run(df = df)
    vi = TestDraw(df)
    fig1 = vi.draw_with_trend(tre_df)
    fig1.write_html("test_ans/trend.html")

    # str_df = StrengthService().run(df= df)
    # vi = TestDraw(df)
    # fig2 = vi.draw_with_strength(str_df)
    # fig2.write_html("test_ans/strength.html")
    