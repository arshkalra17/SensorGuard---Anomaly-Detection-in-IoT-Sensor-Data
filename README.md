# SensorGuard – Anomaly Detection in IoT Sensor Data

Detects abnormal patterns in time-series IoT sensor data using statistical and machine learning methods, then performs basic root cause analysis to identify which sensor triggered the alert.

---

## Problem Statement

IoT devices generate continuous streams of sensor data. Faults — heat spikes, pressure surges, voltage anomalies — can go unnoticed until they cause equipment failure. SensorGuard flags these anomalies in real time and pinpoints the most likely root cause.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| Data | Synthetic IoT dataset (generated in-project) |
| ML Models | Isolation Forest, One-Class SVM (scikit-learn) |
| Statistics | Z-Score (scipy) |
| Feature Eng. | Rolling mean/std, trend, rate-of-change (pandas) |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter |

---

## Project Structure

```
sensorguard/
├── data/
│   └── sensor_data.csv          # generated on first run
├── notebooks/
│   └── SensorGuard_Analysis.ipynb
├── src/
│   ├── generate_data.py         # synthetic dataset generator
│   ├── preprocessing.py         # missing values + normalization
│   ├── feature_engineering.py   # rolling features, trend, ROC
│   ├── anomaly_detection.py     # Z-Score, Isolation Forest, OC-SVM
│   ├── visualization.py         # all plots
│   ├── root_cause.py            # correlation + culprit ranking
│   ├── evaluation.py            # precision, recall, F1
│   └── main.py                  # end-to-end pipeline runner
├── outputs/                     # saved plots + CSVs (auto-created)
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Data Generation
Simulates 2 000 minutes of readings from 5 sensors (temperature, pressure, vibration, humidity, voltage) with realistic noise and ~4 % injected anomalies (ground-truth labels included).

### 2. Preprocessing
- Forward-fill then back-fill missing values
- Min-Max normalization to [0, 1]

### 3. Feature Engineering
For each normalized sensor signal:
- **Rolling mean** (window=10) — smoothed trend
- **Rolling std** — local volatility; high std = unstable sensor
- **Trend (diff)** — rate of change between consecutive readings
- **Rate of change** — percentage change

### 4. Anomaly Detection

| Method | How it works |
|--------|-------------|
| Z-Score | Flags readings > 3 std from the mean |
| Isolation Forest | Isolates outliers via random trees (shorter path = anomaly) |
| One-Class SVM | Learns a boundary around normal data; outside = anomaly |
| Ensemble | Anomaly if flagged by **any** of the three methods |

### 5. Root Cause Analysis
- Pearson correlation heatmap between sensors
- Per-anomaly Z-score per feature → identifies the "top culprit" sensor
- Bar chart ranking sensors by anomaly contribution count

### 6. Evaluation
Compares ensemble predictions against injected ground-truth labels:
- Precision, Recall, F1-Score
- Per-method breakdown
- Confusion matrix

---

## Results (typical run)

| Method | Anomalies Detected |
|--------|--------------------|
| Z-Score | ~3–5 % |
| Isolation Forest | ~5 % (contamination=0.05) |
| One-Class SVM | ~5 % (nu=0.05) |
| Ensemble | ~6–8 % |

Ensemble F1 against ground-truth labels typically lands around **0.55–0.70** depending on random seed — reasonable for unsupervised detection with no tuning.

---

## How to Run

### Option A — Full pipeline (recommended)

```bash
# 1. Clone / open the project
cd sensorguard

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python src/main.py
```

Outputs are saved to `outputs/`:
[Processed Data](outputs/processed_data.csv) — full dataset with all features and anomaly flags
[Anomaly Resulkts](outputs/anomaly_results.csv) — only the anomalous rows
![Sensor Overview](outputs/sensor_overview.png) — time-series with anomalies highlighted
![Anomaly Score](outputs/anomaly_score.png) — Isolation Forest score over time
![Method Comparison](outputs/method_comparison.png) — side-by-side method comparison
![Rolling Features](outputs/rolling_features.png) — rolling mean/std visualization
![Correlation Matrix](outputs/correlation_matrix.png)— feature correlation heatmap
![Root Cause Ranking](outputs/root_cause_ranking.png) — culprit sensor bar chart
![Confusion Matrix](outputs/confusion_matrix.png)— evaluation confusion matrix

### Option B — Jupyter Notebook

```bash
jupyter notebook notebooks/SensorGuard_Analysis.ipynb
```

Run cells top-to-bottom for an interactive walkthrough with inline plots.

---

## Key Concepts (for beginners)

**Z-Score** — measures how many standard deviations a value is from the mean. Beyond ±3 is statistically rare (~0.3% chance in normal data).

**Isolation Forest** — builds random decision trees. Anomalies are "isolated" in fewer splits because they're different from the majority.

**One-Class SVM** — draws a boundary around normal data in high-dimensional space. Anything outside that boundary is flagged.

**Rolling features** — computed over a sliding window of recent readings, capturing local trends rather than global statistics.

---

## Author

Built as a portfolio project demonstrating end-to-end ML engineering on time-series IoT data.
