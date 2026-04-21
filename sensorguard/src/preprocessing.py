"""
preprocessing.py
----------------
Handles loading, cleaning, and normalizing the IoT sensor dataset.

Steps:
  1. Load raw CSV data
  2. Handle missing values
  3. Normalize numeric features to [0, 1] range
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw sensor data from a CSV file."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] Loaded {len(df)} rows from '{filepath}'")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing sensor readings.
    Strategy: forward-fill (use last known value), then back-fill for leading NaNs.
    This is common in time-series because the previous reading is the best estimate.
    """
    before = df.isnull().sum().sum()
    df = df.ffill().bfill()
    after = df.isnull().sum().sum()
    print(f"[INFO] Missing values: {before} → {after} (filled {before - after})")
    return df


def normalize(df: pd.DataFrame, feature_cols: list) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scale each sensor feature to the range [0, 1] using Min-Max scaling.
    Why? ML models (especially Isolation Forest) perform better on normalized data.

    Returns:
        df        : DataFrame with normalized columns (suffixed '_norm')
        scaler    : fitted scaler (save it if you need to inverse-transform later)
    """
    scaler = MinMaxScaler()
    norm_cols = [f"{c}_norm" for c in feature_cols]
    df[norm_cols] = scaler.fit_transform(df[feature_cols])
    print(f"[INFO] Normalized features: {feature_cols}")
    return df, scaler


def preprocess(filepath: str, feature_cols: list) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Full preprocessing pipeline: load → clean → normalize."""
    df = load_data(filepath)
    df = handle_missing_values(df)
    df, scaler = normalize(df, feature_cols)
    return df, scaler


if __name__ == "__main__":
    # Quick smoke-test
    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, "data", "sensor_data.csv")
    features = ["temperature", "pressure", "vibration", "humidity", "voltage"]
    df, scaler = preprocess(path, features)
    print(df.head())
