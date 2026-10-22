import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingestion")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "hourly_vessel_activity.csv"
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "port_congestion_db")

def ingest_batch(new_ais_df: pd.DataFrame, threshold: float = 18.0):
    logger.info(f"Starting ingestion of {len(new_ais_df):,} raw AIS pings...")

    required_cols = ['mmsi', 'base_date_time', 'sog', 'vessel_type']
    for col in required_cols:
        if col not in new_ais_df.columns:
            raise ValueError(f"Missing required AIS column: {col}")

    new_ais_df['base_date_time'] = pd.to_datetime(new_ais_df['base_date_time'])
    new_ais_df = new_ais_df.dropna(subset=['base_date_time', 'mmsi'])
    new_ais_df['timestamp_hour'] = new_ais_df['base_date_time'].dt.floor('h')

    new_ais_df['is_cargo'] = new_ais_df['vessel_type'].between(70, 79).astype(int)
    new_ais_df['is_tanker'] = new_ais_df['vessel_type'].between(80, 89).astype(int)
    new_ais_df['is_tug'] = new_ais_df['vessel_type'].isin([50, 51, 52, 31, 32]).astype(int)
    new_ais_df['is_military'] = new_ais_df['vessel_type'].isin([90, 99, 35, 36, 37]).astype(int)
    new_ais_df['is_low_speed'] = (new_ais_df['sog'] <= 0.5).astype(int)

    new_hourly = new_ais_df.groupby('timestamp_hour').agg(
        vessel_count=('mmsi', 'nunique'),
        ping_count=('sog', 'count'),
        low_speed_vessels=('is_low_speed', 'sum'),
        cargo_vessels=('is_cargo', 'sum'),
        tanker_vessels=('is_tanker', 'sum'),
        tug_vessels=('is_tug', 'sum'),
        military_vessels=('is_military', 'sum'),
        avg_sog=('sog', 'mean'),
        std_sog=('sog', 'std'),
        avg_length=('length', 'mean') if 'length' in new_ais_df.columns else ('mmsi', lambda x: 85.0),
        avg_draft=('draft', 'mean') if 'draft' in new_ais_df.columns else ('mmsi', lambda x: 6.0)
    ).reset_index().rename(columns={'timestamp_hour': 'timestamp'})

    new_hourly['low_speed_ratio'] = np.where(
        new_hourly['vessel_count'] > 0,
        new_hourly['low_speed_vessels'] / new_hourly['vessel_count'],
        0.0
    )

    logger.info(f"Generated {len(new_hourly)} new hourly aggregate records.")

    if DATA_PATH.exists():
        existing_df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
        combined_df = pd.concat([existing_df, new_hourly], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    else:
        combined_df = new_hourly

    for lag in [1, 2, 3, 6, 12, 24]:
        combined_df[f'vessel_count_lag_{lag}h'] = combined_df['vessel_count'].shift(lag)
        combined_df[f'low_speed_lag_{lag}h'] = combined_df['low_speed_vessels'].shift(lag)

    combined_df['vessel_count_diff_1h'] = combined_df['vessel_count'] - combined_df['vessel_count_lag_1h']
    combined_df['vessel_count_diff_24h'] = combined_df['vessel_count'] - combined_df['vessel_count_lag_24h']

    shifted = combined_df['vessel_count'].shift(1)
    combined_df['vessel_count_rolling_mean_6h'] = shifted.rolling(6, min_periods=1).mean()
    combined_df['vessel_count_rolling_std_6h'] = shifted.rolling(6, min_periods=1).std().fillna(0)
    combined_df['vessel_count_rolling_mean_24h'] = shifted.rolling(24, min_periods=1).mean()
    combined_df['vessel_count_rolling_max_24h'] = shifted.rolling(24, min_periods=1).max()

    combined_df['hour'] = combined_df['timestamp'].dt.hour
    combined_df['dayofweek'] = combined_df['timestamp'].dt.dayofweek
    combined_df['is_weekend'] = combined_df['dayofweek'].isin([5, 6]).astype(int)
    combined_df['hour_sin'] = np.sin(2 * np.pi * combined_df['hour'] / 24.0)
    combined_df['hour_cos'] = np.cos(2 * np.pi * combined_df['hour'] / 24.0)
    combined_df['dayofweek_sin'] = np.sin(2 * np.pi * combined_df['dayofweek'] / 7.0)
    combined_df['dayofweek_cos'] = np.cos(2 * np.pi * combined_df['dayofweek'] / 7.0)

    combined_df['target_next_hour_vessel_count'] = combined_df['vessel_count'].shift(-1)
    combined_df['target_high_activity'] = (combined_df['target_next_hour_vessel_count'] >= threshold).astype(float)

    combined_df.to_csv(DATA_PATH, index=False)
    logger.info(f"Updated {DATA_PATH} with total {len(combined_df):,} hourly records.")

    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=1500)
        client.admin.command('ping')
        db = client[MONGODB_DB_NAME]
        coll = db['hourly_vessel_activity']
        
        records = combined_df.tail(len(new_hourly)).to_dict(orient='records')
        for r in records:
            r['timestamp'] = str(r['timestamp'])
            coll.update_one({'timestamp': r['timestamp']}, {'$set': r}, upsert=True)
        logger.info(f"Successfully synced {len(records)} updated records to MongoDB.")
    except Exception as e:
        logger.info(f"MongoDB not reachable ({e}); changes safely stored in local CSV cache.")

    return combined_df

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        logger.info(f"Ingesting file: {path}")
        df = pd.read_csv(path)
        ingest_batch(df)
    else:
        logger.info("ingest.py ready. Usage: python3 scripts/ingest.py <path_to_new_ais_batch.csv>")
