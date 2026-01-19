import pandas as pd
from feature.label.output.label import LabelOutput


class LabelInput:
    def run(self) -> pd.DataFrame:
        return LabelOutput().run()