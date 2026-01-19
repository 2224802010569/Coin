import pandas as pd
from feature.data.usecase.update import Update
from feature.label.entities.participation import Participation
from feature.label.input.data_input import DataInput
from feature.label.service.participation import ParticipationService


class ParticipationUC:
    def run(self) -> pd.DataFrame:
        df = DataInput().run()
        df = ParticipationService().run(df)
        Update().run(entity=Participation,db = df)
        return df