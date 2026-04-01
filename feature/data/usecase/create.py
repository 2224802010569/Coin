from config import DATA

class Create:
    def __init__(self):
        pass

    def run(self, entity = None):
        if entity is None:
            from feature.data.entities.candle import Candle
            entity = Candle
        match DATA.OUTPUT:
            case "sql":
                from feature.data.service.sql import SQLService
                SQLService(entity).create_table()
            # case "pgsql":
            #     from feature.data.service.pgsql import PGSQLService
            #     c = PGSQLService(entity)
