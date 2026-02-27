import pandas as pd
from feature.drl.input.processing_input import ProcessingInput
from feature.drl.input.profile_input import ProfileInput
from feature.drl.service.excution.excution import ExecutionService


class SignUC:
    def __init__(self):
        self.execution = ExecutionService()

    def run(self) -> pd.DataFrame:
        df = ProcessingInput().run()
        profile = ProfileInput().run()
        self.execution.load_signal_model(profile = profile)
        records = []
        for idx in range(len(df)):
            rec = self.execution.run(df, idx, profile)
            if rec is None:
                continue
            records.append({
                "timestamp": rec.timestamp,
                "timeframe": rec.timeframe,
                "label": rec.label,
                "vol": rec.vol,
                "confidence": rec.confidence,
            })
        return pd.DataFrame(records)
