from feature.data.entities.candle import Candle
from feature.data.output.read import DataReadOutput
from feature.data.output.update import DataUpdateOutput


class DataInput:
    @staticmethod
    def read(entity = Candle):
        return DataReadOutput().run(entity)
    
    def update(entity = Candle, db = None):
        return DataUpdateOutput().run(entity , db)