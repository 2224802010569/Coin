import pandas as pd
from feature.data.usecase.read import Read
from feature.data.usecase.create import Create


class DataReadOutput:
    def run(self, entity = None) -> pd.DataFrame:
        df = Read().run(entity)
        if df is None:
            Create().run()
            df = Read().run(entity)
        return df