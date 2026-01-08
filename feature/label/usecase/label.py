from feature.label.entities.strength import Strength
from feature.label.entities.trend import Trend
from feature.label.input.data_input import DataInput
from feature.label.service.strength import StrengthService
from feature.label.service.trend import TrendService


class Label:
    def __init__(self):
        pass

    def run(self):
        df = DataInput().read()
        self.save(Strength, StrengthService().run(df))
        self.save(Trend, TrendService().run(df))

    def save(self, entity= None, df = None):
        if entity is None:
            entity = Trend
        DataInput.update(entity=entity,db = df)