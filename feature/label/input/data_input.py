from feature.data.output.read import DataReadOutput
from feature.data.output.update import DataUpdateOutput
from feature.label.entities.trend import Trend


class DataInput:
    def read(self,entity = None):
        return DataReadOutput().run(entity)
    
    def update(self, entity = Trend, db = None):
        return DataUpdateOutput().run(entity , db)