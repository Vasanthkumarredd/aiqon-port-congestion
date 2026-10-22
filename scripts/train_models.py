import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, classification_report
)
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "hourly_vessel_activity.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def run_training_pipeline():
    print(f"[*] Loading prepared hourly dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])

    feature_cols = [
        'vessel_count',
        'ping_count',
        'low_speed_vessels',
        'cargo_vessels',
        'tanker_vessels',
        'tug_vessels',
        'military_vessels',
        'avg_sog',
        'std_sog',
        'avg_length',
        'avg_draft',
        'low_speed_ratio',
        'vessel_count_lag_1h',
        'low_speed_lag_1h',
        'vessel_count_lag_2h',
        'low_speed_lag_2h',
        'vessel_count_lag_3h',
        'low_speed_lag_3h',
        'vessel_count_lag_6h',
        'low_speed_lag_6h',
        'vessel_count_lag_12h',
        'low_speed_lag_12h',
        'vessel_count_lag_24h',
        'low_speed_lag_24h',
        'vessel_count_diff_1h',
        'vessel_count_diff_24h',
        'vessel_count_rolling_mean_6h',
        'vessel_count_rolling_std_6h',
        'vessel_count_rolling_mean_24h',
        'vessel_count_rolling_max_24h',
        'hour_sin',
        'hour_cos',
        'dayofweek_sin',
        'dayofweek_cos',
        'is_weekend'
    ]

    target_col = 'target_high_activity'

    df = df.sort_values('timestamp').reset_index(drop=True)

    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print(f"[+] Total samples: {len(df):,}")
    print(f"    Train: {len(X_train):,} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"    Test:  {len(X_test):,} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    print(f"    Features count: {len(feature_cols)}")
    print(f"    Train positive rate: {y_train.mean():.3f} | Test positive rate: {y_test.mean():.3f}")

    models = {
        "Dummy_Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic_Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
        ]),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\n[*] Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_proba))
            pr_auc = float(average_precision_score(y_test, y_proba))
            brier = float(brier_score_loss(y_test, y_proba))
        else:
            y_proba = np.zeros_like(y_pred, dtype=float)
            roc_auc = 0.5
            pr_auc = float(y_test.mean())
            brier = float(brier_score_loss(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred).tolist()
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "confusion_matrix": cm,
            "test_samples": len(y_test),
            "true_positives": cm[1][1],
            "false_positives": cm[0][1],
            "true_negatives": cm[0][0],
            "false_negatives": cm[1][0]
        }

        print(f"    Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    rf_model = fitted_models["Random_Forest"]
    rf_importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    xgb_model = fitted_models["XGBoost"]
    xgb_importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    feature_importance_list = []
    feature_descriptions = {
        'vessel_count': 'Current number of active distinct vessels in the port area',
        'vessel_count_lag_1h': 'Vessel count in the previous hour (t-1)',
        'vessel_count_rolling_mean_6h': 'Average vessel activity over the past 6 hours',
        'vessel_count_rolling_mean_24h': 'Average vessel activity over the past 24 hours',
        'vessel_count_diff_1h': 'Short-term momentum: change in vessel count in the last hour',
        'low_speed_vessels': 'Count of stationary/slow vessels (speed <= 0.5 kt, anchored or berthed)',
        'low_speed_ratio': 'Proportion of slow/docked craft relative to total fleet',
        'vessel_count_rolling_max_24h': 'Peak vessel count observed during the last 24 hours',
        'avg_sog': 'Average speed over ground across all active vessels (knots)',
        'cargo_vessels': 'Count of active commercial cargo vessels',
        'tug_vessels': 'Count of active harbor tugs and towing vessels',
        'military_vessels': 'Count of active naval and government vessels',
        'hour_sin': 'Cyclical time of day factor (sine transform)',
        'hour_cos': 'Cyclical time of day factor (cosine transform)',
        'avg_draft': 'Average vessel draft in meters'
    }

    for col in rf_importances.index:
        feature_importance_list.append({
            "feature": col,
            "importance": round(float(rf_importances[col]), 4),
            "xgb_importance": round(float(xgb_importances.get(col, 0.0)), 4),
            "description": feature_descriptions.get(col, f"AIS operational metric: {col.replace('_', ' ')}")
        })

    print("\n=== TOP 10 RANDOM FOREST FEATURES ===")
    for item in feature_importance_list[:10]:
        print(f"  {item['feature']:<30} : {item['importance']:.4f}")

    primary_model_name = "Random_Forest"
    primary_model = fitted_models[primary_model_name]

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_artifact_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump({
        "model": primary_model,
        "feature_cols": feature_cols,
        "model_name": primary_model_name,
        "threshold": 18.0,
        "training_metadata": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "trained_at": pd.Timestamp.now().isoformat()
        }
    }, model_artifact_path)
    print(f"\n[+] Primary model artifact saved to: {model_artifact_path}")

    eval_json_path = os.path.join(MODELS_DIR, "evaluation_results.json")
    with open(eval_json_path, "w") as f:
        json.dump({
            "primary_model": primary_model_name,
            "features": feature_cols,
            "models_comparison": results,
            "top_features": feature_importance_list[:15],
            "all_features_importance": feature_importance_list
        }, f, indent=2)
    print(f"[+] Detailed evaluation results saved to: {eval_json_path}")

    sample_records_path = os.path.join(MODELS_DIR, "sample_feature_records.json")
    cols_to_export = ['timestamp', 'target_high_activity'] + [c for c in feature_cols if c not in ['timestamp', 'target_high_activity']]
    recent_samples = test_df.tail(20)[cols_to_export].to_dict(orient='records')
    for r in recent_samples:
        r['timestamp'] = str(r['timestamp'])
    with open(sample_records_path, "w") as f:
        json.dump(recent_samples, f, indent=2)

    return results

if __name__ == "__main__":
    run_training_pipeline()
