import pandas as pd
from config import DATA

class Read:
    def __init__(self):
        pass

    def run(self, entity = None) -> pd.DataFrame:
        if entity is None:
            from feature.data.entities.candle import Candle
            entity = Candle

        match DATA.OUTPUT:
            case "sql":
                from feature.data.service.sql import SQLService
                return SQLService(entity).read_table()
            # case "pgsql":
            #     from feature.data.service.pgsql import PGSQLService
            #     c = PGSQLService(entity)
