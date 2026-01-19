from feature.data.output.read import DataReadOutput
from feature.data.output.update import DataUpdateOutput
from feature.label.entities.trend import Trend


class DataInput:
    def run(self, tf = "1d"):
        return DataReadOutput().run(tf = tf)