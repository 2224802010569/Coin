from feature.data.entities.candle import Candle
from feature.data.output.read import DataReadOutput
from feature.data.output.update import DataUpdateOutput


class DataInput:
    def run(self):
        return DataReadOutput().run()