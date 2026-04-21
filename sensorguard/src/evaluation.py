"""
evaluation.py
-------------
Evaluates anomaly detection performance against ground-truth labels.

Metrics:
  - Precision : of all flagged anomalies, how many were real?
  - Recall    : of all real anomalies, how many did we catch?
  - F1 Score  : harmonic mean of precision and recall
  - Confusion matrix

Only meaningful when ground-truth labels exist (column 'label' in dataset).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def evaluate(df: pd.DataFrame, pred_col: str = "anomaly", true_col: str = "label"):
    """
    Compare predicted anomalies against ground-truth labels.

    Args:
        df       : DataFrame with both prediction and label columns
        pred_col : column name for predicted anomaly flag (bool/int)
        true_col : column name for ground-truth label (0/1)
    """
    if true_col not in df.columns:
        print(f"[WARN] Column '{true_col}' not found. Skipping evaluation.")
        return

    y_true = df[true_col].astype(int)
    y_pred = df[pred_col].astype(int)

    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "=" * 45)
    print("  EVALUATION RESULTS")
    print("=" * 45)
    print(f"  Precision : {p:.4f}  (fewer false alarms = better)")
    print(f"  Recall    : {r:.4f}  (fewer missed faults = better)")
    print(f"  F1 Score  : {f1:.4f}  (balance of both)")
    print("=" * 45)
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

    # Per-method evaluation
    for method_col in ["zscore_anomaly", "iforest_anomaly", "ocsvm_anomaly"]:
        if method_col in df.columns:
            yp = df[method_col].astype(int)
            mp = precision_score(y_true, yp, zero_division=0)
            mr = recall_score(y_true, yp, zero_division=0)
            mf = f1_score(y_true, yp, zero_division=0)
            print(f"  {method_col:<22} P={mp:.3f}  R={mr:.3f}  F1={mf:.3f}")

    _plot_confusion_matrix(y_true, y_pred)


def _plot_confusion_matrix(y_true, y_pred, save: bool = True):
    """Plot and save the confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Ensemble)", fontweight="bold")
    plt.tight_layout()

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150)
        print(f"[INFO] Saved confusion matrix → {path}")
    plt.close()
