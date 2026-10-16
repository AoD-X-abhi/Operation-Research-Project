"""
Machine Learning Predictive Models
==================================
Predicts:
1. S_{i,k}: Service time (in hours) for ticket k if assigned to resolver i.
2. CSAT_{i,k}: Expected customer satisfaction score (1 to 5).
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CAT_COLS = ['tier', 'resolver', 'priority', 'issue_type', 'channel']
NUM_COLS = ['hour_of_day', 'day_of_week']

def build_pipeline(regressor):
    """Constructs preprocessing + regressor pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), CAT_COLS),
            ('num', 'passthrough', NUM_COLS)
        ]
    )
    return Pipeline([
        ('prep', preprocessor),
        ('reg', regressor)
    ])

def train_and_evaluate_models(data_dir='.'):
    """Trains Service Time and CSAT predictors, evaluates, and saves models."""
    tasks_csv = os.path.join(data_dir, 'data_processed', 'tasks_summary.csv')
    out_dir = os.path.join(data_dir, 'data_processed')
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading task traces from {tasks_csv}...")
    df = pd.read_csv(tasks_csv)
    
    X = df[CAT_COLS + NUM_COLS]
    y_service = df['service_time_h']
    y_csat = df['csat']
    
    X_train, X_test, y_s_train, y_s_test, y_c_train, y_c_test = train_test_split(
        X, y_service, y_csat, test_size=0.2, random_state=42
    )
    
    print("Training Service Time Predictor (Ridge & Random Forest)...")
    service_ridge = build_pipeline(Ridge(alpha=1.0))
    service_ridge.fit(X_train, y_s_train)
    preds_s_ridge = service_ridge.predict(X_test)
    
    service_rf = build_pipeline(RandomForestRegressor(n_estimators=60, max_depth=12, random_state=42, n_jobs=-1))
    service_rf.fit(X_train, y_s_train)
    preds_s_rf = service_rf.predict(X_test)
    
    metrics_service = {
        'Ridge': {
            'MAE': float(mean_absolute_error(y_s_test, preds_s_ridge)),
            'RMSE': float(np.sqrt(mean_squared_error(y_s_test, preds_s_ridge))),
            'R2': float(r2_score(y_s_test, preds_s_ridge))
        },
        'RandomForest': {
            'MAE': float(mean_absolute_error(y_s_test, preds_s_rf)),
            'RMSE': float(np.sqrt(mean_squared_error(y_s_test, preds_s_rf))),
            'R2': float(r2_score(y_s_test, preds_s_rf))
        }
    }
    
    print("Training Customer Satisfaction Predictor (Ridge)...")
    csat_model = build_pipeline(Ridge(alpha=1.0))
    csat_model.fit(X_train, y_c_train)
    preds_c = csat_model.predict(X_test)
    
    metrics_csat = {
        'Ridge': {
            'MAE': float(mean_absolute_error(y_c_test, preds_c)),
            'RMSE': float(np.sqrt(mean_squared_error(y_c_test, preds_c))),
            'R2': float(r2_score(y_c_test, preds_c))
        }
    }
    
    print("\n--- ML EVALUATION RESULTS ---")
    print(f"Service Time (Random Forest) -> MAE: {metrics_service['RandomForest']['MAE']:.3f} h, R2: {metrics_service['RandomForest']['R2']:.4f}")
    print(f"Service Time (Ridge)         -> MAE: {metrics_service['Ridge']['MAE']:.3f} h, R2: {metrics_service['Ridge']['R2']:.4f}")
    print(f"CSAT Score (Ridge)           -> MAE: {metrics_csat['Ridge']['MAE']:.3f}, R2: {metrics_csat['Ridge']['R2']:.4f}")
    
    # Save models
    service_model_path = os.path.join(out_dir, 'service_time_model.joblib')
    csat_model_path = os.path.join(out_dir, 'csat_model.joblib')
    metrics_path = os.path.join(out_dir, 'ml_metrics.json')
    
    joblib.dump(service_rf, service_model_path)
    joblib.dump(csat_model, csat_model_path)
    with open(metrics_path, 'w') as f:
        json.dump({'service_time': metrics_service, 'csat': metrics_csat}, f, indent=2)
        
    print(f"\nModels successfully saved to {out_dir}:")
    print(f"  - {service_model_path}")
    print(f"  - {csat_model_path}")
    print(f"  - {metrics_path}")
    
    return service_rf, csat_model, {'service_time': metrics_service, 'csat': metrics_csat}

class Predictor:
    """Convenience predictor for candidate ticket-resolver assignments."""
    def __init__(self, data_dir='.'):
        out_dir = os.path.join(data_dir, 'data_processed')
        self.service_model = joblib.load(os.path.join(out_dir, 'service_time_model.joblib'))
        self.csat_model = joblib.load(os.path.join(out_dir, 'csat_model.joblib'))
        
    def predict_pair(self, tier, resolver, priority, issue_type, channel, hour=12, dow=2):
        """Predict service time and CSAT for a single candidate pairing."""
        row = pd.DataFrame([{
            'tier': tier,
            'resolver': resolver,
            'priority': priority,
            'issue_type': issue_type,
            'channel': channel,
            'hour_of_day': hour,
            'day_of_week': dow
        }])
        pred_s = max(0.2, float(self.service_model.predict(row)[0]))
        pred_c = float(np.clip(self.csat_model.predict(row)[0], 1.0, 5.0))
        return pred_s, pred_c
        
    def batch_predict(self, pairs_df):
        """Vectorized predictions for multiple candidate pairings."""
        pred_s = np.maximum(0.2, self.service_model.predict(pairs_df))
        pred_c = np.clip(self.csat_model.predict(pairs_df), 1.0, 5.0)
        return pred_s, pred_c

if __name__ == '__main__':
    train_and_evaluate_models()
