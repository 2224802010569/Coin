from feature.label.entities.strength import Strength
from feature.label.entities.trend import Trend
from feature.label.input.data_input import DataInput

class LabelOutput:
    def get_trend(self):
        return DataInput.read(Trend)
    
    def get_strength(self):
        return DataInput.read(Strength)
