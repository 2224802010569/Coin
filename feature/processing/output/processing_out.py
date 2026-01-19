import pandas as pd

from feature.processing.usecase.processing_uc import ProcessingUC

class ProcessingOutput:
    def run(self, tf = "1d") -> pd.DataFrame:
        return ProcessingUC().run(tf = tf)
