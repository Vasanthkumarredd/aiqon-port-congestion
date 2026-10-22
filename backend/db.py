import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import logging
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from backend.config import MONGODB_URI, MONGODB_DB_NAME, HOURLY_CSV_PATH, EVAL_JSON_PATH, SUMMARY_JSON_PATH, DATA_DIR

logger = logging.getLogger("uvicorn")

class Database:
    def __init__(self):
        self.client = None
        self.db = None
        self.is_connected = False
        self._init_connection()

    def _init_connection(self):
        try:
            self.client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
            self.client.admin.command('ping')
            self.db = self.client[MONGODB_DB_NAME]
            self.is_connected = True
            logger.info(f"Connected to MongoDB: {MONGODB_URI} -> {MONGODB_DB_NAME}")
        except Exception as e:
            logger.warning(f"MongoDB not reachable ({e}). Falling back to file cache mode.")
            self.is_connected = False

    def get_summary(self):
        if self.is_connected:
            try:
                meta = self.db.model_metadata.find_one({}, {"_id": 0})
                if meta:
                    return meta
            except Exception as e:
                logger.error(f"Error fetching metadata from Mongo: {e}")

        summary = {}
        if SUMMARY_JSON_PATH.exists():
            with open(SUMMARY_JSON_PATH) as f:
                summary.update(json.load(f))
        if EVAL_JSON_PATH.exists():
            with open(EVAL_JSON_PATH) as f:
                eval_data = json.load(f)
                summary["metrics"] = eval_data.get("models_comparison", {})
                summary["top_features"] = eval_data.get("top_features", [])
        return summary

    def get_activity(self, limit: int = 168, start_date: str = None, end_date: str = None):
        if self.is_connected:
            try:
                query = {}
                if start_date or end_date:
                    query["timestamp"] = {}
                    if start_date:
                        query["timestamp"]["$gte"] = start_date
                    if end_date:
                        query["timestamp"]["$lte"] = end_date

                cursor = self.db.hourly_vessel_activity.find(
                    query, 
                    {"_id": 0, "timestamp": 1, "vessel_count": 1, "ping_count": 1,
                     "low_speed_vessels": 1, "low_speed_ratio": 1, "cargo_vessels": 1,
                     "tanker_vessels": 1, "tug_vessels": 1, "military_vessels": 1,
                     "avg_sog": 1, "target_high_activity": 1}
                ).sort("timestamp", -1).limit(limit)
                
                records = list(cursor)
                records.reverse()
                return records
            except Exception as e:
                logger.error(f"Error querying activity from Mongo: {e}")

        if HOURLY_CSV_PATH.exists():
            df = pd.read_csv(HOURLY_CSV_PATH)
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]
            df = df.tail(limit)
            return df[['timestamp', 'vessel_count', 'ping_count', 'low_speed_vessels', 
                       'low_speed_ratio', 'cargo_vessels', 'tanker_vessels', 'tug_vessels',
                       'military_vessels', 'avg_sog', 'target_high_activity']].to_dict(orient="records")
        return []

    def save_prediction(self, pred_doc: dict):
        doc = pred_doc.copy()
        if self.is_connected:
            try:
                self.db.predictions.insert_one(doc)
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
            except Exception as e:
                logger.error(f"Failed to persist prediction in Mongo: {e}")

        try:
            preds_file = DATA_DIR / "predictions_log.json"
            existing = []
            if preds_file.exists():
                with open(preds_file, "r") as f:
                    try:
                        existing = json.load(f)
                    except Exception:
                        existing = []
            clean_doc = {k: v for k, v in doc.items() if k != "_id"}
            existing.insert(0, clean_doc)
            existing = existing[:200]
            with open(preds_file, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write to local prediction fallback file: {e}")

        return doc

    def get_predictions(self, limit: int = 50):
        if self.is_connected:
            try:
                cursor = self.db.predictions.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
                records = list(cursor)
                if records:
                    return records
            except Exception as e:
                logger.error(f"Error reading predictions from Mongo: {e}")

        preds_file = DATA_DIR / "predictions_log.json"
        if preds_file.exists():
            try:
                with open(preds_file, "r") as f:
                    data = json.load(f)
                    return data[:limit]
            except Exception as e:
                logger.warning(f"Error reading local predictions file: {e}")
        return []

db_client = Database()
