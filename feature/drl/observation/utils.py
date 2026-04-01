import numpy as np
import pandas as pd


def validate_df(df: pd.DataFrame, columns):
    if df is None or df.empty:
        raise ValueError("ObservationProcessor requires a non-empty DataFrame.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing observation columns: {missing}")


def slice_window(df: pd.DataFrame, step: int, window_size: int):
    start = max(0, step - window_size + 1)
    return df.iloc[start: step + 1].copy()


def extract_values(df: pd.DataFrame, columns):
    block = df.loc[:, columns]
    block = block.apply(pd.to_numeric, errors="coerce")
    block = block.replace([np.inf, -np.inf], np.nan)
    block = block.ffill().bfill().fillna(0.0)
    return block.to_numpy(dtype=np.float32)


def normalize_block(values: np.ndarray, use_norm: bool, clip_value: float):
    if not use_norm or values.size == 0:
        return values
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    z = (values - mean) / std
    return np.clip(z, -clip_value, clip_value)


def pad_block(values: np.ndarray, target_rows: int):
    if len(values) >= target_rows:
        return values
    pad = target_rows - len(values)
    return np.pad(values, ((pad, 0), (0, 0)), mode="edge")


def build_extra_state(account_state, extra_size: int):
    if account_state is None:
        return np.zeros(extra_size, dtype=np.float32)
    arr = np.asarray(account_state, dtype=np.float32).reshape(-1)
    if len(arr) >= extra_size:
        return arr[:extra_size]
    return np.pad(arr, (0, extra_size - len(arr)), mode="constant")