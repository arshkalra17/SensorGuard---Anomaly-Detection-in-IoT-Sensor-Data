"""
root_cause.py
-------------
Performs basic root cause analysis on detected anomalies.

Approach:
  1. Correlation matrix  : which features move together?
  2. Feature contribution: for each anomaly, which feature deviated most?
  3. Top culprit summary : rank features by how often they caused anomalies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


# ─── Correlation Analysis ─────────────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame, feature_cols: list, save: bool = True):
    """
    Heatmap of Pearson correlations between sensor features.
    High correlation between two features means they tend to rise/fall together.
    If both spike during an anomaly, they may share a root cause.
    """
    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
        plt.savefig(path, dpi=150)
        print(f"[INFO] Saved correlation matrix → {path}")
    plt.close()
    return corr


# ─── Feature Contribution ─────────────────────────────────────────────────────

def feature_contribution(df: pd.DataFrame, raw_feature_cols: list) -> pd.DataFrame:
    """
    For each anomaly row, compute how many standard deviations each raw feature
    is away from its mean (absolute Z-score).

    The feature with the highest Z-score in that row is the 'top culprit'.
    """
    anomalies = df[df["anomaly"] == True].copy()
    if anomalies.empty:
        print("[WARN] No anomalies found for root cause analysis.")
        return pd.DataFrame()

    # Z-score of raw features across the full dataset
    means = df[raw_feature_cols].mean()
    stds  = df[raw_feature_cols].std().replace(0, 1)  # avoid division by zero

    z = (anomalies[raw_feature_cols] - means) / stds
    z = z.abs()

    anomalies["top_culprit"] = z.idxmax(axis=1)
    anomalies["max_deviation"] = z.max(axis=1)

    print(f"[INFO] Root cause analysis on {len(anomalies)} anomalies.")
    return anomalies[["timestamp"] + raw_feature_cols + ["top_culprit", "max_deviation", "anomaly"]]


# ─── Culprit Ranking ──────────────────────────────────────────────────────────

def rank_culprits(anomaly_df: pd.DataFrame, save: bool = True) -> pd.Series:
    """
    Count how many times each feature was the 'top culprit'.
    This tells you which sensor is most responsible for anomalies overall.
    """
    if anomaly_df.empty or "top_culprit" not in anomaly_df.columns:
        return pd.Series(dtype=int)

    counts = anomaly_df["top_culprit"].value_counts()
    print("\n[Root Cause Ranking]")
    print(counts.to_string())

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color="tomato", edgecolor="black")
    ax.set_title("Root Cause: Feature Anomaly Contribution Count", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sensor Feature")
    ax.set_ylabel("Number of Anomalies Caused")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "root_cause_ranking.png")
        plt.savefig(path, dpi=150)
        print(f"[INFO] Saved root cause ranking → {path}")
    plt.close()
    return counts


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_root_cause_analysis(df: pd.DataFrame, raw_feature_cols: list):
    """Full root cause analysis pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Correlation
    corr = plot_correlation_matrix(df, raw_feature_cols)

    # 2. Per-anomaly contribution
    anomaly_df = feature_contribution(df, raw_feature_cols)

    # 3. Ranking
    counts = rank_culprits(anomaly_df)

    # 4. Simple text summary
    if not counts.empty:
        top = counts.index[0]
        print(f"\n[Summary] The most common root cause is '{top}' "
              f"({counts.iloc[0]} anomalies). "
              f"Check this sensor first when investigating alerts.")

    return corr, anomaly_df, counts


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import preprocess
    from feature_engineering import engineer_features
    from anomaly_detection import detect_anomalies

    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, "data", "sensor_data.csv")
    raw_features = ["temperature", "pressure", "vibration", "humidity", "voltage"]
    norm_cols = [f"{c}_norm" for c in raw_features]

    df, _ = preprocess(path, raw_features)
    df = engineer_features(df, norm_cols)
    eng_cols = [c for c in df.columns if any(c.endswith(s) for s in
                ["_norm", "_roll_mean", "_roll_std", "_trend", "_roc"])]
    df = detect_anomalies(df, eng_cols)
    run_root_cause_analysis(df, raw_features)
