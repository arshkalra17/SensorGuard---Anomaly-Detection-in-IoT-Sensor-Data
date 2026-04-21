"""
anomaly_detection.py
--------------------
Detects anomalies using three methods:

  1. Z-Score       : flags readings that are statistically far from the mean
  2. Isolation Forest : tree-based ML model that isolates outliers
  3. One-Class SVM : learns the boundary of "normal" data (bonus)

Each method adds a boolean column to the DataFrame:
  - 'zscore_anomaly'
  - 'iforest_anomaly'
  - 'ocsvm_anomaly'

A final 'anomaly' column is True if ANY method flags the row.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


# ─── Z-Score ──────────────────────────────────────────────────────────────────

def zscore_detection(df: pd.DataFrame, cols: list, threshold: float = 3.0) -> pd.DataFrame:
    """
    Z-Score method:
      z = (x - mean) / std
      If |z| > threshold (default 3), the point is an anomaly.

    Why 3? In a normal distribution, 99.7% of data falls within ±3 std.
    Anything beyond that is statistically unusual.
    """
    z_scores = np.abs(stats.zscore(df[cols].fillna(0)))
    # Flag a row if ANY feature has |z| > threshold
    df["zscore_anomaly"] = (z_scores > threshold).any(axis=1)
    n = df["zscore_anomaly"].sum()
    print(f"[Z-Score]  Anomalies detected: {n} ({n/len(df)*100:.2f}%)")
    return df


# ─── Isolation Forest ─────────────────────────────────────────────────────────

def isolation_forest_detection(
    df: pd.DataFrame,
    cols: list,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Isolation Forest:
      Builds random trees that try to 'isolate' each data point.
      Anomalies are isolated faster (fewer splits needed) → shorter path length.

    contamination: expected fraction of anomalies in the dataset (5% here).
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    preds = model.fit_predict(df[cols].fillna(0))
    # IsolationForest returns -1 for anomalies, 1 for normal
    df["iforest_anomaly"] = preds == -1
    df["iforest_score"] = model.decision_function(df[cols].fillna(0))  # lower = more anomalous
    n = df["iforest_anomaly"].sum()
    print(f"[IForest]  Anomalies detected: {n} ({n/len(df)*100:.2f}%)")
    return df


# ─── One-Class SVM (Bonus) ────────────────────────────────────────────────────

def ocsvm_detection(
    df: pd.DataFrame,
    cols: list,
    nu: float = 0.05,
) -> pd.DataFrame:
    """
    One-Class SVM:
      Learns a tight boundary around 'normal' data in high-dimensional space.
      Points outside the boundary are anomalies.

    nu: upper bound on the fraction of outliers (similar to contamination).
    Note: Slower than Isolation Forest on large datasets.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols].fillna(0))

    model = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    preds = model.fit_predict(X)
    # OneClassSVM returns -1 for anomalies, 1 for normal
    df["ocsvm_anomaly"] = preds == -1
    n = df["ocsvm_anomaly"].sum()
    print(f"[OC-SVM]   Anomalies detected: {n} ({n/len(df)*100:.2f}%)")
    return df


# ─── Ensemble ─────────────────────────────────────────────────────────────────

def combine_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all three methods: a row is an anomaly if flagged by ANY method.
    You can change this to require 2-of-3 for stricter detection.
    """
    cols = [c for c in ["zscore_anomaly", "iforest_anomaly", "ocsvm_anomaly"] if c in df.columns]
    df["anomaly"] = df[cols].any(axis=1)
    total = df["anomaly"].sum()
    print(f"[Ensemble] Total anomalies (any method): {total} ({total/len(df)*100:.2f}%)")
    return df


# ─── Main pipeline ────────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Run all detection methods and combine results.
    `feature_cols` should be the engineered feature columns.
    """
    df = zscore_detection(df, feature_cols)
    df = isolation_forest_detection(df, feature_cols)
    df = ocsvm_detection(df, feature_cols)
    df = combine_anomalies(df)
    return df


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import preprocess
    from feature_engineering import engineer_features

    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, "data", "sensor_data.csv")
    raw_features = ["temperature", "pressure", "vibration", "humidity", "voltage"]
    norm_cols = [f"{c}_norm" for c in raw_features]

    df, _ = preprocess(path, raw_features)
    df = engineer_features(df, norm_cols)

    # Use all engineered numeric columns as input to detectors
    eng_cols = [c for c in df.columns if any(c.endswith(s) for s in
                ["_norm", "_roll_mean", "_roll_std", "_trend", "_roc"])]
    df = detect_anomalies(df, eng_cols)
    print(df[["timestamp", "anomaly", "zscore_anomaly", "iforest_anomaly", "ocsvm_anomaly"]].tail(20))
