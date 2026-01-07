import pandas as pd
from feature.data.usecase.read import Read
from feature.data.usecase.create import Create
from feature.data.usecase.update import Update


class DataUpdateOutput:
    def run(self, entity = None, db: pd.DataFrame = None):
        df = Read().run(entity)
        if df is None:
            Create().run()
        
        Update().run(entity = entity, db = db)