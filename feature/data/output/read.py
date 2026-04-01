import pandas as pd
from feature.data.entities.candle import Candle
from feature.data.usecase.profile import ProfileUC
from feature.data.usecase.read import Read
from feature.data.usecase.create import Create
from feature.data.usecase.update import Update


class DataReadOutput:
    def run(self, tf = "1d") -> pd.DataFrame:
        df = Read().run(Candle, tf = tf)
        if df is None or df.empty:
            Create().run(Candle)
            Update().run(Candle)
            ProfileUC().run()
            df = Read().run(Candle, tf = tf)
        return df
    
class ProfileOutput:
    def run (self, name: str = "balanced") -> pd.DataFrame:
        df = ProfileUC().run()
        df = df.loc[df['name']==name]
        return df