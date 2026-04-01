import pandas as pd
from config import PROCESSING


class SplitService:
    def __init__(self):
        self.train = PROCESSING.TRAIN
        self.test = PROCESSING.TEST
        self.step =  PROCESSING.STEP

    def run(self, df: pd.DataFrame = None):
        split = []
        for i in range(0,len(df) - self.train - self.test+1 , self.step):
            train_df = df.iloc[i: i+self.train]
            test_df = df.iloc[i+self.train : i + self.train + self.test]
            split.append((train_df,test_df))
        return split