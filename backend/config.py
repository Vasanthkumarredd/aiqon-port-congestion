import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "port_congestion_db")
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
HOURLY_CSV_PATH = DATA_DIR / "hourly_vessel_activity.csv"
MODEL_JOB_PATH = MODELS_DIR / "best_model.joblib"
EVAL_JSON_PATH = MODELS_DIR / "evaluation_results.json"
SUMMARY_JSON_PATH = DATA_DIR / "dataset_summary.json"
