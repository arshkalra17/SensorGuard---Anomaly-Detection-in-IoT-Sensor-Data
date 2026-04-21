"""
generate_data.py
----------------
Generates a realistic synthetic IoT sensor dataset and saves it to data/sensor_data.csv.

Sensors simulated:
  - temperature  (°C)  : base ~25°C with daily cycle
  - pressure     (hPa) : base ~1013 hPa
  - vibration    (g)   : base ~0.5g, spiky
  - humidity     (%)   : base ~50%
  - voltage      (V)   : base ~220V

Anomalies are injected at random intervals to simulate real faults.
The 'label' column marks ground-truth anomalies (useful for evaluation).
"""

import numpy as np
import pandas as pd
import os

SEED = 42
N_POINTS = 2000          # total time steps (≈ 33 hours at 1-min intervals)
ANOMALY_RATE = 0.04      # ~4% of points are true anomalies


def generate(n: int = N_POINTS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Time axis ──────────────────────────────────────────────────────────────
    timestamps = pd.date_range(start="2024-01-01 00:00", periods=n, freq="1min")

    # ── Normal sensor signals ──────────────────────────────────────────────────
    t = np.linspace(0, 4 * np.pi, n)   # for sinusoidal patterns

    temperature = 25 + 5 * np.sin(t) + rng.normal(0, 0.5, n)
    pressure    = 1013 + 3 * np.cos(t * 0.5) + rng.normal(0, 0.8, n)
    vibration   = 0.5 + 0.1 * np.abs(np.sin(t * 2)) + rng.exponential(0.05, n)
    humidity    = 50 + 8 * np.sin(t * 0.3 + 1) + rng.normal(0, 1.5, n)
    voltage     = 220 + 2 * np.sin(t * 0.8) + rng.normal(0, 1.0, n)

    # ── Inject anomalies ───────────────────────────────────────────────────────
    n_anomalies = int(n * ANOMALY_RATE)
    anomaly_idx = rng.choice(n, size=n_anomalies, replace=False)
    labels = np.zeros(n, dtype=int)

    for idx in anomaly_idx:
        labels[idx] = 1
        # Randomly pick which sensor(s) spike
        spike_type = rng.integers(0, 5)
        if spike_type == 0:
            temperature[idx] += rng.uniform(15, 25)   # heat spike
        elif spike_type == 1:
            pressure[idx] += rng.uniform(30, 60)      # pressure surge
        elif spike_type == 2:
            vibration[idx] += rng.uniform(2, 5)       # mechanical fault
        elif spike_type == 3:
            humidity[idx] += rng.uniform(30, 45)      # moisture intrusion
        else:
            voltage[idx] += rng.uniform(20, 50)       # voltage surge

    # ── Introduce a few missing values (realistic) ─────────────────────────────
    for arr in [temperature, pressure, vibration, humidity, voltage]:
        missing_idx = rng.choice(n, size=int(n * 0.005), replace=False)
        arr[missing_idx] = np.nan

    df = pd.DataFrame({
        "timestamp":   timestamps,
        "temperature": np.round(temperature, 3),
        "pressure":    np.round(pressure, 3),
        "vibration":   np.round(vibration, 4),
        "humidity":    np.round(humidity, 3),
        "voltage":     np.round(voltage, 3),
        "label":       labels,   # 1 = true anomaly (ground truth)
    })

    return df


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    out_path = os.path.join(base, "data", "sensor_data.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = generate()
    df.to_csv(out_path, index=False)
    print(f"[INFO] Dataset saved → {out_path}")
    print(f"       Rows: {len(df)} | Anomalies (ground truth): {df['label'].sum()}")
    print(df.head())
