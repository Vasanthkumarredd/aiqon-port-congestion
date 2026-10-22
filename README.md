# Aiqon AI — Port Congestion Risk Prediction Engine

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0+-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end Machine Learning and full-stack operational intelligence platform built for the **Aiqon AI Internship Assignment: Predicting Port Congestion and Shipment Delays Using AI**.

The system ingests and processes **3.24 million AIS records** from the Port of Guam (Apra Harbor) for the entire year of 2025, defines a defensible operational congestion proxy, benchmarks baseline and tree ensemble models via strict chronological evaluation, and serves real-time predictions via a FastAPI backend, MongoDB database, and a responsive React (Vite + Recharts) dashboard.

---

## Architecture Overview

```
                         ┌──────────────────────────────────────────────┐
                         │   Raw AIS Dataset (guam_2025.csv, 3.24M)     │
                         └──────────────────────┬───────────────────────┘
                                                │ (Chunked Processing)
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │  Data Preparation & Feature Engineering      │
                         │  - Hourly aggregation (MMSIs, SOG, Types)    │
                         │  - 35 Lag, Rolling & Cyclical Features       │
                         │  - Defensible Congestion Proxy Target (>=18) │
                         └──────────────────────┬───────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
┌─────────────────────────────────┐                           ┌─────────────────────────────────┐
│     Model Training Pipeline     │                           │   Storage & Caching Tier        │
│  - Dummy Baseline (70.4% Acc)   │                           │   - data/hourly_activity.csv    │
│  - Logistic Regression (87.8%)  │                           │   - MongoDB (PyMongo)           │
│  - Random Forest (89.9% Acc) ★  │                           │     • hourly_vessel_activity    │
│  - XGBoost (90.8% Acc)          │                           │     • predictions               │
│  - Chronological Test Set Split │                           │     • model_metadata            │
│  - Artifacts in models/         │                           └────────────────┬────────────────┘
└────────────────┬────────────────┘                                            │
                 │                                                             │
                 ▼                                                             ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                FastAPI REST Backend (:8000)                                   │
│   Endpoints: /health, /api/dashboard/summary, /api/activity, /api/predict, /api/predictions   │
└───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                │ REST API (JSON)
                                                ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                               React + Vite Web Dashboard (:3000)                              │
│   KPIs, Recharts Hourly Trend, Live Risk Predictor, Model Benchmarks, Feature Explainability   │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Defensible Prediction Target & Scientific Notice

> **Crucial Data Integrity Rule:** The raw AIS dataset does **not** contain explicit shipment-delay or customs clearance records. In accordance with sound machine learning practice, this project does **not** claim to predict unverified shipment delays.
>
> Instead, we construct a data-grounded, operationally valid target: **Next-Hour High Vessel Activity as a Port Congestion Proxy**.
> 
> The threshold ($\ge 18.0$ active vessels) was derived **strictly from the 75th percentile of the chronological training period** (Jan 1 to Oct 1, 2025) to guarantee **zero future-data leakage**.

---

## Machine Learning Benchmark Results

Evaluated on chronologically held-out test data (**October 1 – December 31, 2025: $N=2,184$ hours**):

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** | 70.38% | 0.00% | 0.00% | 0.00% | 0.5000 | 0.2962 |
| **Logistic Regression** | 87.77% | 75.00% | **88.10%** | 81.02% | 0.9549 | 0.0914 |
| **Random Forest (Primary)** | **89.93%** | **80.54%** | 87.02% | **83.66%** | **0.9627** | **0.0718** |
| **XGBoost (Comparison)** | 90.80% | 84.31% | 84.70% | 84.50% | 0.9640 | 0.0712 |

### Top Predictive Features
1. `vessel_count` (22.89%): Real-time port vessel density.
2. `ping_count` (12.86%): Frequency of AIS telemetry transmissions.
3. `vessel_count_lag_1h` (9.17%): Previous hour's vessel count.
4. `low_speed_vessels` (6.11%): Stationary/docked craft ($SOG \le 0.5$ kt) indicating berth occupancy.
5. `vessel_count_lag_2h` (5.93%): Multi-hour persistence.

---

## Quickstart & Setup Guide

### 1. Prerequisites
- **Python 3.10+**
- **Node.js 18+** & npm
- **MongoDB** (running locally on port 27017, optional with automatic file fallback)

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data & Engineer Features
Chunk-processes the 3.24M raw records and generates 8,735 hourly records:
```bash
python3 scripts/prepare_data.py
```

### 4. Train & Evaluate Models
Trains all 4 models, tests chronologically, and saves artifacts to `models/`:
```bash
python3 scripts/train_models.py
```

### 5. Seed MongoDB
Populates `hourly_vessel_activity`, `model_metadata`, and `predictions`:
```bash
python3 scripts/seed_db.py
```

### 6. Launch FastAPI Backend
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
- Interactive Swagger API docs: `http://localhost:8000/docs`
- Healthcheck: `http://localhost:8000/health`

### 7. Launch React Web Dashboard
In a separate terminal:
```bash
cd frontend
npm install
npm run dev -- --port 3000
```
- Web Dashboard: `http://localhost:3000`

---

## FastAPI REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Healthcheck, MongoDB status, and model loaded state |
| `GET` | `/api/dashboard/summary` | Aggregated dataset summary, time coverage, current risk snapshot |
| `GET` | `/api/activity` | Historical hourly vessel activity (supports `limit`, `start_date`, `end_date`) |
| `POST` | `/api/predict` | Next-hour congestion risk inference with local explainability |
| `GET` | `/api/predictions` | Recent model prediction history stored in MongoDB |
| `GET` | `/api/model/metrics` | Benchmark metrics (Baseline vs LR vs RF vs XGBoost, Confusion Matrix) |
| `GET` | `/api/model/feature-importance` | Ranked feature importance weights and plain-English descriptions |

### Sample Inference Request (`POST /api/predict`)
```json
{
  "vessel_count": 22,
  "low_speed_vessels": 14,
  "avg_sog": 1.8,
  "cargo_vessels": 6,
  "tug_vessels": 6,
  "military_vessels": 4
}
```

### Sample Inference Response
```json
{
  "risk_class": 1,
  "risk_label": "High Congestion Risk",
  "risk_probability": 0.8897,
  "confidence_level": "High",
  "threshold": 18.0,
  "timestamp": "2026-10-21 12:01:52",
  "explanations": [
    {
      "feature": "Current Vessel Count",
      "impact": "Positive (+)",
      "description": "Current count (22) meets or exceeds the congestion proxy threshold (18 vessels)."
    },
    {
      "feature": "Stationary / Low-Speed Ratio",
      "impact": "Positive (+)",
      "description": "63.6% of vessels are stationary/docked, indicating high berth occupancy."
    }
  ]
}
```

---

## Project Structure

```
aiqon-port-congestion/
├── data/
│   ├── guam_2025.csv                  # Symlink to raw AIS dataset
│   ├── hourly_vessel_activity.csv     # 8,735 prepared hourly records
│   └── dataset_summary.json           # Time bounds, thresholds, counts
├── scripts/
│   ├── prepare_data.py                # Chunked data pipeline & feature engineering
│   ├── train_models.py                # Chronological training & benchmark evaluation
│   └── seed_db.py                     # MongoDB database seeder
├── models/
│   ├── best_model.joblib              # Fitted Random Forest artifact
│   └── evaluation_results.json        # Test scores, confusion matrix, features
├── backend/
│   ├── app.py                         # FastAPI web service & prediction routes
│   ├── config.py                      # Environment and paths configuration
│   ├── db.py                          # Resilient MongoDB client & file cache fallback
│   └── schemas.py                     # Pydantic validation models
├── frontend/
│   ├── index.html                     # Web entry point
│   ├── package.json                   # Dependencies (React 18, Vite, Recharts)
│   ├── vite.config.js                 # Vite dev server config
│   └── src/
│       ├── main.jsx                   # Application bootstrap
│       ├── App.jsx                    # Multi-tab responsive dashboard
│       └── index.css                  # Marine dark-mode design system
├── requirements.txt                   # Python requirements
└── README.md                          # Project documentation
```
