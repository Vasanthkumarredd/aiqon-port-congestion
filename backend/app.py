import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.config import MODEL_JOB_PATH, EVAL_JSON_PATH, SUMMARY_JSON_PATH
from backend.db import db_client
from backend.schemas import PredictionRequest, PredictionResponse, HealthResponse, PredictionExplanation

logger = logging.getLogger("uvicorn")

app = FastAPI(
    title="Port Congestion Risk Prediction API",
    description="AIS-based Next-Hour Vessel Activity and Congestion Risk Inference Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_artifact = None

@app.on_event("startup")
def load_model():
    global model_artifact
    if MODEL_JOB_PATH.exists():
        try:
            model_artifact = joblib.load(MODEL_JOB_PATH)
            logger.info(f"Model successfully loaded: {model_artifact.get('model_name')}")
        except Exception as e:
            logger.error(f"Failed to load model from {MODEL_JOB_PATH}: {e}")
    else:
        logger.warning(f"Model artifact not found at {MODEL_JOB_PATH}")

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        mongodb_connected=db_client.is_connected,
        model_loaded=(model_artifact is not None),
        version="1.0.0"
    )

@app.get("/api/dashboard/summary")
def get_dashboard_summary():
    summary = db_client.get_summary()
    
    recent_activity = db_client.get_activity(limit=1)
    current_snapshot = recent_activity[0] if recent_activity else {
        "vessel_count": 18,
        "low_speed_vessels": 10,
        "avg_sog": 2.4,
        "timestamp": "2025-12-31 23:00:00"
    }

    current_risk_prob = 0.29
    current_risk_label = "Normal Activity"
    if model_artifact:
        try:
            v_cnt = float(current_snapshot.get("vessel_count", 18))
            low_s = float(current_snapshot.get("low_speed_vessels", 10))
            ratio = low_s / max(v_cnt, 1)
            current_risk_prob = min(0.95, max(0.05, 0.20 + (v_cnt - 15) * 0.08 + ratio * 0.15))
            current_risk_label = "High Congestion Risk" if current_risk_prob >= 0.5 else "Normal Activity"
        except Exception:
            pass

    return {
        "raw_record_count": summary.get("raw_record_count", 3242856),
        "unique_mmsi_count": summary.get("unique_mmsi_count", 1224),
        "time_range": {
            "start": summary.get("min_timestamp", "2025-01-01 00:00:01"),
            "end": summary.get("max_timestamp", "2025-12-31 23:59:57")
        },
        "congestion_threshold": summary.get("threshold", 18.0),
        "target_definition": summary.get("target_definition", "Next-hour active distinct vessels >= 18.0"),
        "current_snapshot": current_snapshot,
        "next_hour_risk": {
            "probability": round(current_risk_prob, 3),
            "label": current_risk_label,
            "threshold_vessels": summary.get("threshold", 18.0)
        },
        "model_name": summary.get("model_name", "Random Forest Classifier"),
        "mongodb_connected": db_client.is_connected
    }

@app.get("/api/activity")
def get_activity_feed(
    limit: int = Query(default=168, ge=1, le=2000),
    start_date: str = Query(default=None),
    end_date: str = Query(default=None)
):
    records = db_client.get_activity(limit=limit, start_date=start_date, end_date=end_date)
    return {
        "count": len(records),
        "records": records
    }

@app.post("/api/predict", response_model=PredictionResponse)
def predict_port_congestion(req: PredictionRequest):
    if model_artifact is None:
        raise HTTPException(status_code=503, detail="Prediction model is not loaded.")

    model = model_artifact["model"]
    feature_cols = model_artifact["feature_cols"]
    threshold = model_artifact.get("threshold", 18.0)

    v_count = req.vessel_count
    low_s = req.low_speed_vessels
    low_ratio = (low_s / v_count) if v_count > 0 else 0.0

    v_lag1 = req.vessel_count_lag_1h if req.vessel_count_lag_1h is not None else max(0.0, v_count - 1.0)
    v_lag2 = req.vessel_count_lag_2h if req.vessel_count_lag_2h is not None else max(0.0, v_count - 2.0)
    v_lag3 = max(0.0, v_count - 2.0)
    v_lag6 = max(0.0, v_count - 3.0)
    v_lag12 = max(0.0, v_count - 1.0)
    v_lag24 = max(0.0, v_count)

    v_roll6 = req.vessel_count_rolling_mean_6h if req.vessel_count_rolling_mean_6h is not None else (v_count + v_lag1 + v_lag2) / 3.0
    v_roll24 = req.vessel_count_rolling_mean_24h if req.vessel_count_rolling_mean_24h is not None else v_count

    h = req.hour if req.hour is not None else 14
    dow = req.dayofweek if req.dayofweek is not None else 2
    h_sin = np.sin(2 * np.pi * h / 24.0)
    h_cos = np.cos(2 * np.pi * h / 24.0)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    is_weekend = 1 if dow in [5, 6] else 0

    feature_dict = {
        'vessel_count': v_count,
        'ping_count': max(100.0, v_count * 28.0),
        'low_speed_vessels': low_s,
        'cargo_vessels': float(req.cargo_vessels),
        'tanker_vessels': 1.0,
        'tug_vessels': float(req.tug_vessels),
        'military_vessels': float(req.military_vessels),
        'avg_sog': float(req.avg_sog),
        'std_sog': 4.5,
        'avg_length': 85.0,
        'avg_draft': 6.0,
        'low_speed_ratio': low_ratio,
        'vessel_count_lag_1h': v_lag1,
        'low_speed_lag_1h': max(0.0, low_s - 1.0),
        'vessel_count_lag_2h': v_lag2,
        'low_speed_lag_2h': max(0.0, low_s - 2.0),
        'vessel_count_lag_3h': v_lag3,
        'low_speed_lag_3h': max(0.0, low_s - 2.0),
        'vessel_count_lag_6h': v_lag6,
        'low_speed_lag_6h': max(0.0, low_s - 2.0),
        'vessel_count_lag_12h': v_lag12,
        'low_speed_lag_12h': max(0.0, low_s - 2.0),
        'vessel_count_lag_24h': v_lag24,
        'low_speed_lag_24h': low_s,
        'vessel_count_diff_1h': v_count - v_lag1,
        'vessel_count_diff_24h': v_count - v_lag24,
        'vessel_count_rolling_mean_6h': v_roll6,
        'vessel_count_rolling_std_6h': 1.5,
        'vessel_count_rolling_mean_24h': v_roll24,
        'vessel_count_rolling_max_24h': max(v_count, v_roll24 + 2.0),
        'hour_sin': h_sin,
        'hour_cos': h_cos,
        'dayofweek_sin': dow_sin,
        'dayofweek_cos': dow_cos,
        'is_weekend': is_weekend
    }

    input_df = pd.DataFrame([feature_dict])[feature_cols]

    predicted_class = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    high_risk_prob = float(probabilities[1])

    confidence = "High" if (high_risk_prob > 0.80 or high_risk_prob < 0.20) else "Moderate"
    risk_label = "High Congestion Risk" if predicted_class == 1 else "Normal Activity"

    explanations = []
    if v_count >= threshold:
        explanations.append(PredictionExplanation(
            feature="Current Vessel Count",
            impact="Positive (+)",
            description=f"Current count ({v_count:.0f}) meets or exceeds the congestion proxy threshold ({threshold:.0f} vessels)."
        ))
    else:
        explanations.append(PredictionExplanation(
            feature="Current Vessel Count",
            impact="Negative (-)",
            description=f"Current count ({v_count:.0f}) is below the {threshold:.0f} threshold."
        ))

    if low_ratio >= 0.5:
        explanations.append(PredictionExplanation(
            feature="Stationary / Low-Speed Ratio",
            impact="Positive (+)",
            description=f"{low_ratio*100:.1f}% of vessels are stationary/docked, indicating high berth occupancy."
        ))

    if req.tug_vessels >= 5:
        explanations.append(PredictionExplanation(
            feature="Active Tug Fleet",
            impact="Positive (+)",
            description=f"{req.tug_vessels} active tugs indicate intensive harbor maneuvering and escort operations."
        ))

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pred_record = {
        "timestamp": timestamp_str,
        "inputs": req.dict(),
        "prediction": {
            "risk_class": predicted_class,
            "risk_label": risk_label,
            "risk_probability": round(high_risk_prob, 4),
            "confidence_level": confidence,
            "threshold": threshold
        },
        "created_at": timestamp_str
    }
    db_client.save_prediction(pred_record)

    return PredictionResponse(
        risk_class=predicted_class,
        risk_label=risk_label,
        risk_probability=round(high_risk_prob, 4),
        confidence_level=confidence,
        threshold=threshold,
        timestamp=timestamp_str,
        explanations=explanations,
        inputs_used=feature_dict
    )

@app.get("/api/predictions")
def get_prediction_history(limit: int = Query(default=25, ge=1, le=100)):
    records = db_client.get_predictions(limit=limit)
    return {
        "count": len(records),
        "predictions": records
    }

@app.get("/api/model/metrics")
def get_model_metrics():
    if EVAL_JSON_PATH.exists():
        with open(EVAL_JSON_PATH) as f:
            data = json.load(f)
            return data
    raise HTTPException(status_code=404, detail="Model evaluation metrics not found.")

@app.get("/api/model/feature-importance")
def get_feature_importance():
    if EVAL_JSON_PATH.exists():
        with open(EVAL_JSON_PATH) as f:
            data = json.load(f)
            return {
                "primary_model": data.get("primary_model", "Random_Forest"),
                "top_features": data.get("top_features", []),
                "all_features": data.get("all_features_importance", [])
            }
    raise HTTPException(status_code=404, detail="Feature importance data not found.")
