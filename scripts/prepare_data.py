import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json

RAW_CSV_PATH = os.environ.get("AIS_CSV_PATH", "/Users/vasanth/Downloads/guam_2025.csv")
OUTPUT_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hourly_vessel_activity.csv")
SUMMARY_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset_summary.json")

def process_ais_data(input_path: str, chunk_size: int = 250000):
    print(f"[*] Starting AIS data processing from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source file not found at: {input_path}")

    usecols = [
        'mmsi', 'base_date_time', 'sog', 'vessel_type', 
        'length', 'width', 'draft', 'status'
    ]
    dtypes = {
        'mmsi': 'int64',
        'sog': 'float32',
        'vessel_type': 'float32',
        'length': 'float32',
        'width': 'float32',
        'draft': 'float32',
        'status': 'float32'
    }

    hour_mmsi_stats = {}
    total_raw_rows = 0
    all_unique_mmsis = set()
    min_timestamp = None
    max_timestamp = None

    print(f"[*] Reading dataset in chunks of {chunk_size:,} rows...")
    chunk_idx = 0
    for chunk in pd.read_csv(input_path, usecols=usecols, dtype=dtypes, 
                             parse_dates=['base_date_time'], chunksize=chunk_size):
        chunk_idx += 1
        total_raw_rows += len(chunk)

        chunk = chunk.dropna(subset=['base_date_time', 'mmsi'])
        if len(chunk) == 0:
            continue

        c_min = chunk['base_date_time'].min()
        c_max = chunk['base_date_time'].max()
        min_timestamp = c_min if min_timestamp is None else min(min_timestamp, c_min)
        max_timestamp = c_max if max_timestamp is None else max(max_timestamp, c_max)

        chunk['timestamp_hour'] = chunk['base_date_time'].dt.floor('h')
        all_unique_mmsis.update(chunk['mmsi'].unique().tolist())

        grouped = chunk.groupby(['timestamp_hour', 'mmsi']).agg(
            pings=('sog', 'count'),
            mean_sog=('sog', 'mean'),
            vessel_type=('vessel_type', 'first'),
            length=('length', 'first'),
            draft=('draft', 'first')
        ).reset_index()

        for row in grouped.itertuples(index=False):
            key = (row.timestamp_hour, row.mmsi)
            if key not in hour_mmsi_stats:
                hour_mmsi_stats[key] = {
                    'pings': row.pings,
                    'sog_sum': (row.mean_sog if not np.isnan(row.mean_sog) else 0.0) * row.pings,
                    'vessel_type': row.vessel_type,
                    'length': row.length,
                    'draft': row.draft
                }
            else:
                existing = hour_mmsi_stats[key]
                existing['pings'] += row.pings
                existing['sog_sum'] += (row.mean_sog if not np.isnan(row.mean_sog) else 0.0) * row.pings
                if np.isnan(existing['vessel_type']) and not np.isnan(row.vessel_type):
                    existing['vessel_type'] = row.vessel_type
                if np.isnan(existing['length']) and not np.isnan(row.length):
                    existing['length'] = row.length
                if np.isnan(existing['draft']) and not np.isnan(row.draft):
                    existing['draft'] = row.draft

        print(f"    - Processed chunk {chunk_idx}: {total_raw_rows:,} records processed so far...")

    print(f"\n[+] Raw processing complete:")
    print(f"    Total records: {total_raw_rows:,}")
    print(f"    Unique MMSIs: {len(all_unique_mmsis):,}")
    print(f"    Time range: {min_timestamp} to {max_timestamp}")

    records = []
    for (t_hour, mmsi), data in hour_mmsi_stats.items():
        avg_speed = (data['sog_sum'] / data['pings']) if data['pings'] > 0 else 0.0
        records.append({
            'timestamp': t_hour,
            'mmsi': mmsi,
            'pings': data['pings'],
            'avg_sog': avg_speed,
            'vessel_type': data['vessel_type'],
            'length': data['length'],
            'draft': data['draft']
        })

    vessel_hour_df = pd.DataFrame(records)
    del hour_mmsi_stats

    print("[*] Aggregating hourly operational metrics...")
    
    vessel_hour_df['is_cargo'] = vessel_hour_df['vessel_type'].between(70, 79).astype(int)
    vessel_hour_df['is_tanker'] = vessel_hour_df['vessel_type'].between(80, 89).astype(int)
    vessel_hour_df['is_tug'] = vessel_hour_df['vessel_type'].isin([50, 51, 52, 31, 32]).astype(int)
    vessel_hour_df['is_military_gov'] = vessel_hour_df['vessel_type'].isin([90, 99, 35, 36, 37]).astype(int)
    vessel_hour_df['is_low_speed'] = (vessel_hour_df['avg_sog'] <= 0.5).astype(int)

    hourly_df = vessel_hour_df.groupby('timestamp').agg(
        vessel_count=('mmsi', 'nunique'),
        ping_count=('pings', 'sum'),
        low_speed_vessels=('is_low_speed', 'sum'),
        cargo_vessels=('is_cargo', 'sum'),
        tanker_vessels=('is_tanker', 'sum'),
        tug_vessels=('is_tug', 'sum'),
        military_vessels=('is_military_gov', 'sum'),
        avg_sog=('avg_sog', 'mean'),
        std_sog=('avg_sog', 'std'),
        avg_length=('length', 'mean'),
        avg_draft=('draft', 'mean')
    ).reset_index()

    full_hourly_idx = pd.date_range(
        start=hourly_df['timestamp'].min(),
        end=hourly_df['timestamp'].max(),
        freq='h',
        name='timestamp'
    )
    hourly_df = hourly_df.set_index('timestamp').reindex(full_hourly_idx).reset_index()

    count_cols = ['vessel_count', 'ping_count', 'low_speed_vessels', 
                  'cargo_vessels', 'tanker_vessels', 'tug_vessels', 'military_vessels']
    hourly_df[count_cols] = hourly_df[count_cols].fillna(0)

    hourly_df['avg_sog'] = hourly_df['avg_sog'].fillna(0.0)
    hourly_df['std_sog'] = hourly_df['std_sog'].fillna(0.0)
    hourly_df['avg_length'] = hourly_df['avg_length'].ffill().bfill()
    hourly_df['avg_draft'] = hourly_df['avg_draft'].ffill().bfill()

    hourly_df['low_speed_ratio'] = np.where(
        hourly_df['vessel_count'] > 0,
        hourly_df['low_speed_vessels'] / hourly_df['vessel_count'],
        0.0
    )

    print(f"[+] Continuous time series generated: {len(hourly_df)} hours.")

    print("[*] Generating lag and rolling features...")
    
    for lag in [1, 2, 3, 6, 12, 24]:
        hourly_df[f'vessel_count_lag_{lag}h'] = hourly_df['vessel_count'].shift(lag)
        hourly_df[f'low_speed_lag_{lag}h'] = hourly_df['low_speed_vessels'].shift(lag)
    
    hourly_df['vessel_count_diff_1h'] = hourly_df['vessel_count'] - hourly_df['vessel_count_lag_1h']
    hourly_df['vessel_count_diff_24h'] = hourly_df['vessel_count'] - hourly_df['vessel_count_lag_24h']

    shifted_vessel_count = hourly_df['vessel_count'].shift(1)
    hourly_df['vessel_count_rolling_mean_6h'] = shifted_vessel_count.rolling(window=6, min_periods=1).mean()
    hourly_df['vessel_count_rolling_std_6h'] = shifted_vessel_count.rolling(window=6, min_periods=1).std().fillna(0)
    hourly_df['vessel_count_rolling_mean_24h'] = shifted_vessel_count.rolling(window=24, min_periods=1).mean()
    hourly_df['vessel_count_rolling_max_24h'] = shifted_vessel_count.rolling(window=24, min_periods=1).max()

    hourly_df['hour'] = hourly_df['timestamp'].dt.hour
    hourly_df['dayofweek'] = hourly_df['timestamp'].dt.dayofweek
    hourly_df['is_weekend'] = hourly_df['dayofweek'].isin([5, 6]).astype(int)
    hourly_df['month'] = hourly_df['timestamp'].dt.month

    hourly_df['hour_sin'] = np.sin(2 * np.pi * hourly_df['hour'] / 24.0)
    hourly_df['hour_cos'] = np.cos(2 * np.pi * hourly_df['hour'] / 24.0)
    hourly_df['dayofweek_sin'] = np.sin(2 * np.pi * hourly_df['dayofweek'] / 7.0)
    hourly_df['dayofweek_cos'] = np.cos(2 * np.pi * hourly_df['dayofweek'] / 7.0)

    hourly_df['target_next_hour_vessel_count'] = hourly_df['vessel_count'].shift(-1)

    valid_df = hourly_df.dropna(subset=['vessel_count_lag_24h', 'target_next_hour_vessel_count']).copy()
    valid_df['target_next_hour_vessel_count'] = valid_df['target_next_hour_vessel_count'].astype(int)

    split_idx = int(len(valid_df) * 0.75)
    train_slice = valid_df.iloc[:split_idx]
    test_slice = valid_df.iloc[split_idx:]

    high_activity_threshold = float(train_slice['vessel_count'].quantile(0.75))
    print(f"\n[+] Congestion Proxy Threshold (Calculated STRICTLY from Training Period, 75th percentile):")
    print(f"    Threshold: >= {high_activity_threshold:.1f} vessels active in an hour")

    valid_df['target_high_activity'] = (valid_df['target_next_hour_vessel_count'] >= high_activity_threshold).astype(int)

    train_high_pct = (valid_df.iloc[:split_idx]['target_high_activity'].mean()) * 100
    test_high_pct = (valid_df.iloc[split_idx:]['target_high_activity'].mean()) * 100
    print(f"    Training set positive rate: {train_high_pct:.1f}%")
    print(f"    Test set positive rate:     {test_high_pct:.1f}%")

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    valid_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n[+] Successfully saved prepared hourly dataset to:\n    {OUTPUT_CSV_PATH}")
    print(f"    Total hourly records: {len(valid_df):,}")

    summary = {
        "raw_record_count": int(total_raw_rows),
        "unique_mmsi_count": int(len(all_unique_mmsis)),
        "min_timestamp": str(min_timestamp),
        "max_timestamp": str(max_timestamp),
        "total_hourly_records": int(len(valid_df)),
        "train_records": int(len(train_slice)),
        "test_records": int(len(test_slice)),
        "train_start": str(train_slice['timestamp'].min()),
        "train_end": str(train_slice['timestamp'].max()),
        "test_start": str(test_slice['timestamp'].min()),
        "test_end": str(test_slice['timestamp'].max()),
        "high_activity_threshold": high_activity_threshold,
        "target_column": "target_high_activity",
        "proxy_definition": f"Next-hour active distinct vessels >= {high_activity_threshold:.1f} (training set 75th percentile)"
    }
    with open(SUMMARY_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved dataset summary to {SUMMARY_JSON_PATH}")

    return valid_df, summary

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else RAW_CSV_PATH
    process_ais_data(csv_file)
