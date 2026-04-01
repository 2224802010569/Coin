from feature.data.output.read import DataReadOutput

class DataInput:
    def run(self, tf = "1d"):
        return DataReadOutput().run(tf = tf)