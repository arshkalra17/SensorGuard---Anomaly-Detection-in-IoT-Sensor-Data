"""
main.py
-------
End-to-end pipeline for SensorGuard.

Run this file to:
  1. Generate (or load) sensor data
  2. Preprocess
  3. Engineer features
  4. Detect anomalies (Z-Score + Isolation Forest + One-Class SVM)
  5. Visualize results
  6. Root cause analysis
  7. Evaluate against ground-truth labels
  8. Save outputs
"""

import os
import sys

# Make sure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from generate_data      import generate
from preprocessing      import preprocess
from feature_engineering import engineer_features
from anomaly_detection  import detect_anomalies
from visualization      import run_all_plots
from root_cause         import run_root_cause_analysis
from evaluation         import evaluate

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data",    "sensor_data.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
RAW_FEATURES = ["temperature", "pressure", "vibration", "humidity", "voltage"]
NORM_COLS    = [f"{c}_norm" for c in RAW_FEATURES]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Generate data if not present ──────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print("[Step 1] Generating synthetic sensor dataset...")
        df_raw = generate()
        df_raw.to_csv(DATA_PATH, index=False)
        print(f"         Saved to {DATA_PATH}\n")
    else:
        print(f"[Step 1] Data already exists at {DATA_PATH}\n")

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    print("[Step 2] Preprocessing...")
    df, scaler = preprocess(DATA_PATH, RAW_FEATURES)
    print()

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    print("[Step 3] Engineering features...")
    df = engineer_features(df, NORM_COLS)
    print()

    # ── Step 4: Anomaly Detection ─────────────────────────────────────────────
    print("[Step 4] Detecting anomalies...")
    # Use all engineered numeric columns as model input
    eng_cols = [c for c in df.columns if any(
        c.endswith(s) for s in ["_norm", "_roll_mean", "_roll_std", "_trend", "_roc"]
    )]
    df = detect_anomalies(df, eng_cols)
    print()

    # ── Step 5: Visualize ─────────────────────────────────────────────────────
    print("[Step 5] Generating plots...")
    run_all_plots(df, RAW_FEATURES)
    print()

    # ── Step 6: Root Cause Analysis ───────────────────────────────────────────
    print("[Step 6] Running root cause analysis...")
    run_root_cause_analysis(df, RAW_FEATURES)
    print()

    # ── Step 7: Evaluation ────────────────────────────────────────────────────
    print("[Step 7] Evaluating against ground-truth labels...")
    evaluate(df)
    print()

    # ── Step 8: Save outputs ──────────────────────────────────────────────────
    print("[Step 8] Saving results...")
    processed_path = os.path.join(OUTPUT_DIR, "processed_data.csv")
    anomaly_path   = os.path.join(OUTPUT_DIR, "anomaly_results.csv")

    df.to_csv(processed_path, index=False)
    df[df["anomaly"] == True].to_csv(anomaly_path, index=False)

    print(f"         Processed dataset → {processed_path}")
    print(f"         Anomaly rows only → {anomaly_path}")
    print("\n✅ SensorGuard pipeline complete. Check the outputs/ folder.")


if __name__ == "__main__":
    main()
