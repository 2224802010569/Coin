from feature.data.output.read import DataReadOutput
from feature.data.output.update import DataUpdateOutput
from feature.label.entities.trend import Trend


class DataInput:
    def read(entity = Trend):
        return DataReadOutput().run(entity)
    
    def update(entity = Trend, db = None):
        return DataUpdateOutput().run(entity , db)