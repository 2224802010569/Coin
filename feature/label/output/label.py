from feature.label.entities.strength import Strength
from feature.label.entities.trend import Trend
from feature.label.input.data_input import DataInput
from feature.label.usecase.label import Label

class LabelOutput:
    def read_trend(self):
        try:
            return DataInput.read(entity = Trend)
        except:
            Label().run()
            return DataInput.read(entity = Trend)
    
    def read_strength(self):
        try:
            return DataInput.read(entity = Strength)
        except:
            Label().run()
            return DataInput.read(entity = Strength)
