import pandas as pd
from feature.data.usecase.update import Update
from feature.label.entities.trend import Trend
from feature.label.input.data_input import DataInput
from feature.label.service.trend import TrendService


class TrendUC:
    def run(self) -> pd.DataFrame:
        df = DataInput().run()
        df = TrendService().run(df)
        Update().run(entity=Trend,db = df)
        return df