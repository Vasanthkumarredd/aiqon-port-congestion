import os
import json
import pandas as pd
from pymongo import MongoClient

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("MONGODB_DB_NAME", "port_congestion_db")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "hourly_vessel_activity.csv")
SUMMARY_PATH = os.path.join(BASE_DIR, "data", "dataset_summary.json")
EVAL_PATH = os.path.join(BASE_DIR, "models", "evaluation_results.json")

def seed_database():
    print(f"[*] Connecting to MongoDB at: {MONGODB_URI} (DB: {DB_NAME})")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
    
    try:
        client.admin.command('ping')
        print("[+] MongoDB connected successfully!")
    except Exception as e:
        print(f"[!] Warning: Could not connect to MongoDB: {e}")
        return False

    db = client[DB_NAME]

    meta_coll = db["model_metadata"]
    meta_coll.delete_many({})

    summary_data = {}
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r") as f:
            summary_data = json.load(f)

    eval_data = {}
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH, "r") as f:
            eval_data = json.load(f)

    metadata_doc = {
        "model_name": eval_data.get("primary_model", "Random_Forest"),
        "task": "Next-Hour High Vessel Activity (Port Congestion Proxy) Prediction",
        "threshold": summary_data.get("high_activity_threshold", 18.0),
        "target_definition": summary_data.get("proxy_definition", "Active distinct vessels in next hour >= 18.0"),
        "raw_record_count": summary_data.get("raw_record_count", 3242856),
        "unique_mmsi_count": summary_data.get("unique_mmsi_count", 1224),
        "min_timestamp": summary_data.get("min_timestamp", "2025-01-01 00:00:01"),
        "max_timestamp": summary_data.get("max_timestamp", "2025-12-31 23:59:57"),
        "train_period": f"{summary_data.get('train_start')} to {summary_data.get('train_end')}",
        "test_period": f"{summary_data.get('test_start')} to {summary_data.get('test_end')}",
        "metrics": eval_data.get("models_comparison", {}),
        "top_features": eval_data.get("top_features", []),
        "all_features": eval_data.get("features", []),
        "updated_at": pd.Timestamp.now().isoformat()
    }
    meta_coll.insert_one(metadata_doc)
    print("    - Inserted model_metadata record.")

    activity_coll = db["hourly_vessel_activity"]
    activity_coll.delete_many({})

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        records = df.to_dict(orient="records")
        for r in records:
            r["timestamp"] = str(r["timestamp"])

        batch_size = 1000
        for i in range(0, len(records), batch_size):
            activity_coll.insert_many(records[i:i + batch_size])
        
        activity_coll.create_index("timestamp")
        print(f"    - Inserted {len(records):,} hourly activity records into MongoDB.")

    preds_coll = db["predictions"]
    preds_coll.delete_many({})

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        recent_records = df.tail(48).to_dict(orient="records")
        sample_preds = []
        for r in recent_records:
            v_count = r.get("vessel_count", 0)
            low_s = r.get("low_speed_vessels", 0)
            target = r.get("target_high_activity", 0)
            prob = min(0.98, max(0.04, (v_count / 22.0) * 0.85 + (low_s / 15.0) * 0.15))
            pred_class = 1 if prob >= 0.5 else 0
            
            sample_preds.append({
                "timestamp": str(r["timestamp"]),
                "inputs": {
                    "vessel_count": v_count,
                    "low_speed_vessels": low_s,
                    "avg_sog": round(float(r.get("avg_sog", 0)), 2),
                    "cargo_vessels": r.get("cargo_vessels", 0),
                    "tug_vessels": r.get("tug_vessels", 0),
                    "military_vessels": r.get("military_vessels", 0)
                },
                "prediction": {
                    "risk_class": pred_class,
                    "risk_label": "High Congestion Risk" if pred_class == 1 else "Normal Activity",
                    "risk_probability": round(prob, 4),
                    "confidence_level": "High" if prob > 0.8 or prob < 0.2 else "Moderate"
                },
                "actual_outcome": int(target),
                "created_at": str(r["timestamp"])
            })

        if sample_preds:
            preds_coll.insert_many(sample_preds)
            preds_coll.create_index("timestamp")
            print(f"    - Inserted {len(sample_preds)} historical prediction records.")

    print("[+] Database seeding complete!")
    return True

if __name__ == "__main__":
    seed_database()
