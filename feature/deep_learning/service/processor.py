import pandas as pd
import numpy as np

class ProcessorService:
    def __init__(self):
        self.mean = None
        self.std = None
        self.feature_cols = None

    def _infer_features(self, df: pd.DataFrame):
        return df.select_dtypes(include=["number"]).columns.tolist()

    def fit(self, df: pd.DataFrame):
        self.feature_cols = self._infer_features(df)
        x = df[self.feature_cols].astype(float)
        self.mean = x.mean()
        self.std = x.std().replace(0, 1.0)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_cols].astype(float)
        x = (x - self.mean) / self.std
        return x.to_numpy(dtype=np.float32)

    def run(self, list_split):
        train_df, test_df = list_split
        self.fit(train_df)
        x_train = self.transform(train_df)
        x_test = self.transform(test_df)
        return x_train, x_test