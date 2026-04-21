"""
visualization.py
----------------
Plots time-series sensor data and highlights detected anomalies.
All plots are saved to the outputs/ folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
sns.set_theme(style="darkgrid", palette="muted")


def plot_sensor_overview(df: pd.DataFrame, raw_cols: list, save: bool = True):
    """
    Multi-panel time-series plot: one subplot per sensor.
    Anomaly points are highlighted in red.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n = len(raw_cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    fig.suptitle("SensorGuard – IoT Sensor Readings with Anomalies", fontsize=15, fontweight="bold")

    for ax, col in zip(axes, raw_cols):
        # Normal readings
        ax.plot(df["timestamp"], df[col], color="steelblue", linewidth=0.8, label=col)

        # Overlay anomaly points
        anom = df[df["anomaly"] == True]
        ax.scatter(anom["timestamp"], anom[col], color="red", s=20, zorder=5, label="Anomaly")

        ax.set_ylabel(col, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "sensor_overview.png")
        plt.savefig(path, dpi=150)
        print(f"[INFO] Saved sensor overview → {path}")
    plt.close()


def plot_anomaly_score(df: pd.DataFrame, save: bool = True):
    """
    Plot the Isolation Forest anomaly score over time.
    Lower score = more anomalous.
    """
    if "iforest_score" not in df.columns:
        print("[WARN] iforest_score not found, skipping score plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["timestamp"], df["iforest_score"], color="darkorange", linewidth=0.8, label="IF Score")
    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="Decision boundary")

    anom = df[df["anomaly"] == True]
    ax.scatter(anom["timestamp"], anom["iforest_score"], color="red", s=20, zorder=5)

    ax.set_title("Isolation Forest Anomaly Score Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Anomaly Score (lower = more anomalous)")
    ax.legend()
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "anomaly_score.png")
        plt.savefig(path, dpi=150)
        print(f"[INFO] Saved anomaly score plot → {path}")
    plt.close()


def plot_method_comparison(df: pd.DataFrame, col: str = "temperature", save: bool = True):
    """
    Side-by-side comparison of which method flagged anomalies for one sensor.
    """
    methods = {
        "Z-Score": "zscore_anomaly",
        "Isolation Forest": "iforest_anomaly",
        "One-Class SVM": "ocsvm_anomaly",
    }
    available = {k: v for k, v in methods.items() if v in df.columns}
    n = len(available)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Anomaly Detection Method Comparison – {col}", fontsize=13, fontweight="bold")

    for ax, (method_name, flag_col) in zip(axes, available.items()):
        ax.plot(df["timestamp"], df[col], color="steelblue", linewidth=0.8)
        anom = df[df[flag_col] == True]
        ax.scatter(anom["timestamp"], anom[col], color="red", s=18, zorder=5, label=f"{method_name} anomaly")
        ax.set_ylabel(col)
        ax.set_title(method_name, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "method_comparison.png")
        plt.savefig(path, dpi=150)
        print(f"[INFO] Saved method comparison → {path}")
    plt.close()


def plot_rolling_features(df: pd.DataFrame, col: str = "temperature_norm", save: bool = True):
    """
    Show raw normalized signal alongside its rolling mean and rolling std.
    Helps visualize what the feature engineering step produced.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(f"Rolling Features – {col}", fontsize=13, fontweight="bold")

    ax1.plot(df["timestamp"], df[col], color="steelblue", linewidth=0.7, label="Normalized")
    if f"{col}_roll_mean" in df.columns:
        ax1.plot(df["timestamp"], df[f"{col}_roll_mean"], color="orange", linewidth=1.2, label="Rolling Mean")
    ax1.set_ylabel("Value")
    ax1.legend()

    if f"{col}_roll_std" in df.columns:
        ax2.plot(df["timestamp"], df[f"{col}_roll_std"], color="green", linewidth=0.8, label="Rolling Std")
        ax2.set_ylabel("Std Dev")
        ax2.legend()

    axes_list = [ax1, ax2]
    axes_list[-1].set_xlabel("Time")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "rolling_features.png")
        plt.savefig(path, dpi=150)
        print(f"[INFO] Saved rolling features plot → {path}")
    plt.close()


def run_all_plots(df: pd.DataFrame, raw_cols: list):
    """Generate all visualizations."""
    plot_sensor_overview(df, raw_cols)
    plot_anomaly_score(df)
    plot_method_comparison(df, col=raw_cols[0])
    plot_rolling_features(df, col=f"{raw_cols[0]}_norm")
    print("[INFO] All plots saved to outputs/")
