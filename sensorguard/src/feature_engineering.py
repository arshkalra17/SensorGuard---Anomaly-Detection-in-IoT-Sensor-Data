"""
feature_engineering.py
-----------------------
Creates time-series features from normalized sensor readings.

New features:
  - Rolling mean   : smoothed signal (captures trend)
  - Rolling std    : local volatility (spikes = anomaly candidate)
  - Trend (diff)   : rate of change between consecutive readings
  - Rate of change : percentage change
"""

import pandas as pd
import numpy as np


WINDOW = 10   # number of time steps for rolling calculations


def add_rolling_mean(df: pd.DataFrame, cols: list, window: int = WINDOW) -> pd.DataFrame:
    """
    Rolling mean over `window` steps.
    Smooths out noise so we can see the underlying trend.
    """
    for col in cols:
        df[f"{col}_roll_mean"] = df[col].rolling(window=window, min_periods=1).mean()
    print(f"[INFO] Added rolling mean (window={window}) for {cols}")
    return df


def add_rolling_std(df: pd.DataFrame, cols: list, window: int = WINDOW) -> pd.DataFrame:
    """
    Rolling standard deviation over `window` steps.
    High std → the sensor is fluctuating a lot → potential anomaly.
    """
    for col in cols:
        df[f"{col}_roll_std"] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
    print(f"[INFO] Added rolling std (window={window}) for {cols}")
    return df


def add_trend(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    First-order difference: value[t] - value[t-1].
    Captures how fast the sensor reading is changing.
    """
    for col in cols:
        df[f"{col}_trend"] = df[col].diff().fillna(0)
    print(f"[INFO] Added trend (diff) for {cols}")
    return df


def add_rate_of_change(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Percentage change: (value[t] - value[t-1]) / value[t-1].
    Useful for detecting sudden relative jumps.
    """
    for col in cols:
        df[f"{col}_roc"] = df[col].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
    print(f"[INFO] Added rate-of-change for {cols}")
    return df


def engineer_features(df: pd.DataFrame, norm_cols: list) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Expects normalized columns (e.g. 'temperature_norm').
    """
    df = add_rolling_mean(df, norm_cols)
    df = add_rolling_std(df, norm_cols)
    df = add_trend(df, norm_cols)
    df = add_rate_of_change(df, norm_cols)
    print(f"[INFO] Feature engineering complete. Total columns: {len(df.columns)}")
    return df


if __name__ == "__main__":
    import os
    from preprocessing import preprocess

    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, "data", "sensor_data.csv")
    features = ["temperature", "pressure", "vibration", "humidity", "voltage"]
    norm_cols = [f"{c}_norm" for c in features]

    df, _ = preprocess(path, features)
    df = engineer_features(df, norm_cols)
    print(df.head())
