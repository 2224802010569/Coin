from feature.deep_learning.entities.recommand import Recommand
from feature.deep_learning.input.data_input import DataInput


class DeepOutput:
    def read_recommand(self):
        try:
            return DataInput.read(entity = Recommand)
        except:
            # Label().run()
            return DataInput.read(entity = Recommand)