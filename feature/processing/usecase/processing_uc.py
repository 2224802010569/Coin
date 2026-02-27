import pandas as pd
from feature.data.usecase.read import Read
from feature.data.usecase.update import Update
from feature.processing.input.data_input import DataInput
from feature.processing.entities.processing import Processing
from feature.processing.service.processing import ProcessingService
from config import COIN

class ProcessingUC:
    def __init__(self):
        pass

    def run(self, tf: str = "1d") -> pd.DataFrame:
        df = Read().run(entity = Processing, tf = tf)
        if df is None or df.empty:
            self.make()
            df = Read().run(entity = Processing, tf = tf)
        return df
    
    def make(self):
        for tf in COIN.TIMEFRAME:
            df = DataInput().run(tf = tf)
            df = ProcessingService().run(df = df)
            Update().run(entity = Processing, db = df)
