import pandas as pd
from config import DATA

class Update:
    def __init__(self):
        pass

    def run(self, entity = None, db: pd.DataFrame = None):
        if entity is None:
            from feature.data.entities.candle import Candle
            entity = Candle
        if db is None:
            from feature.data.service.input import InputService
            db = InputService().run()
        match DATA.OUTPUT:
            case "sql":
                from feature.data.service.sql import SQLService
                SQLService(entity).update_table(db)
            # case "pgsql":
            #     from feature.data.service.pgsql import PGSQLService
            #     c = PGSQLService(entity)
