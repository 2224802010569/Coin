import pandas as pd
from feature.processing.output.processing_out import ProcessingOutput


class ProcessingInput:
    def run(self, tf ="1d") -> pd.DataFrame:
        return ProcessingOutput().run(tf = tf)