from feature.label.entities.strength import Strength
from feature.label.entities.trend import Trend
from feature.label.input.data_input import DataInput
from feature.label.service.strength import strength
from feature.label.service.trend import trend


class label:
    def __init__(self):
        pass

    def run(self):
        df = DataInput().read()
        self.save(Strength, strength().run(df))
        self.save(Trend, trend().run(df))
        pass

    def save(self, entity= None, df = None):
        if entity is None:
            entity = Trend
        DataInput.update(entity,df)