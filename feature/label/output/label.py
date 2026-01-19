import pandas as pd
from feature.label.usecase.label_uc import LabelUC

class LabelOutput:
    def run(self) -> pd.DataFrame:
        return LabelUC().run()
