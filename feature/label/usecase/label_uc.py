import pandas as pd
from feature.data.entities.candle import Candle
from feature.label.entities.participation import Participation
from feature.label.entities.trend import Trend
from feature.label.input.data_input import DataInput
from feature.label.service.participation import ParticipationService
from feature.label.service.sign import SignService
from feature.label.service.trend import TrendService
from feature.label.usecase.participation_uc import ParticipationUC
from feature.label.usecase.trend_uc import TrendUC


class LabelUC:
    def __init__(self):
        pass

    def run(self) -> pd.DataFrame:
        df = DataInput().run()
        trend =  TrendUC().run()
        sign = SignService().map_with_candle(candle = df, df = trend)
        sign = SignService().sign_for_trend(df = sign)
        part = ParticipationUC().run()
        return SignService().map_all(trend = sign , participation = part)
