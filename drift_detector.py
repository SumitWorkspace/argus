import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

"""
What is Data Drift?
-------------------
Data drift occurs when the statistical properties of the model's input features change over time 
compared to the data the model was trained on. Since machine learning models make predictions under 
the assumption that the future will resemble the past, significant data drift can lead to model 
degradation, where the model's predictive accuracy drops in production.

By periodically monitoring and detecting data drift, we can trigger alerts for:
1. Model retraining: Updating the model with newer representative data.
2. Feature engineering: Adjusting data scaling or handling new patterns.
3. System debugging: Detecting upstream data pipeline ingestion bugs.
"""

def compute_reference_stats():
    """
    Computes summary statistics for baseline numerical features ('Time' and 'Amount') 
    from the original training dataset (creditcard.csv) and saves them to 'reference_stats.json'.
    
    Why this check matters:
    -----------------------
    This establishes the ground-truth historical distribution of our features. It stores 
    statistical parameters (mean, standard deviation, min, max, percentiles) so we can 
    compare production requests against them without having to read the massive creditcard.csv 
    file on every health check or request.
    """
    reference_file = 'reference_stats.json'
    csv_path = 'creditcard.csv'
    
    # If stats are already computed, load them directly (useful for docker/production environments)
    if os.path.exists(reference_file):
        print(f"Reference stats file '{reference_file}' already exists. Loading statistics...")
        with open(reference_file, 'r') as f:
            return json.load(f)
            
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Original dataset '{csv_path}' not found. Cannot compute baseline stats.")
        
    print(f"Computing baseline reference stats from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    
    stats = {}
    for col in ['Time', 'Amount']:
        series = df[col]
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "percentiles": {
                "25": float(series.quantile(0.25)),
                "50": float(series.quantile(0.50)),
                "75": float(series.quantile(0.75))
            }
        }
        
    with open(reference_file, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Saved baseline reference stats to '{reference_file}'.")
    return stats


def detect_drift(n_recent=100):
    """
    Compares the distribution of the last N predictions against baseline training statistics.
    
    Check 1: Mean Shift (Time and Amount)
    -----------------------------------
    Formula: abs(recent_mean - reference_mean) > 2 * reference_std
    
    Why it matters:
    - Amount Shift: If the average transaction amount shifts significantly (e.g. from $88 to $400),
      it indicates that users are performing atypical purchases. The model might not generalize well 
      to these high-value transactions.
    - Time Shift: A shift in transaction times could mean transactions are occurring at odd hours 
      (e.g., late-night spikes), which is a common indicator of automated fraudulent activities.
    
    Check 2: Fraud Rate Shift
    -------------------------
    Formula: recent_fraud_rate > 3 * reference_fraud_rate (0.173%)
    
    Why it matters:
    - A massive spike in predicted fraud rate (e.g., jumping from 0.17% to 5%) suggests either:
      a) A coordinated cyberattack/fraud campaign is active.
      b) The model is suffering from 'concept drift' and over-predicting fraud (high false positives),
         damaging user experience by declining legitimate transactions.
    """
    log_path = 'predictions_log.jsonl'
    reference_file = 'reference_stats.json'
    
    # 1. Ensure reference stats exist
    if not os.path.exists(reference_file):
        try:
            ref_stats = compute_reference_stats()
        except Exception as e:
            return {
                "error": f"Baseline statistics file '{reference_file}' is missing and cannot be computed: {str(e)}",
                "drift_detected": False,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "sample_size": 0,
                "alerts": [],
                "fraud_rate": {"reference": 0.00173, "recent": 0.0, "flagged": False}
            }
    else:
        with open(reference_file, 'r') as f:
            ref_stats = json.load(f)
            
    # 2. Read recent predictions from log file
    recent_transactions = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()
            last_lines = lines[-n_recent:] if len(lines) > n_recent else lines
            for line in last_lines:
                if line.strip():
                    try:
                        recent_transactions.append(json.loads(line))
                    except Exception:
                        pass
                        
    sample_size = len(recent_transactions)
    checked_at = datetime.now(timezone.utc).isoformat()
    
    if sample_size == 0:
        return {
            "drift_detected": False,
            "checked_at": checked_at,
            "sample_size": 0,
            "alerts": [],
            "fraud_rate": {
                "reference": 0.00173,
                "recent": 0.0,
                "flagged": False
            },
            "message": "No predictions logged yet. Drift detection requires at least 1 prediction log."
        }
        
    df_recent = pd.DataFrame(recent_transactions)
    alerts = []
    drift_detected = False
    
    # 3. Perform Check 1: Mean Shift on Time and Amount
    # Note: Log fields are lowercase ("time", "amount") while reference stats are capitalized
    for feature_lower, feature_ref in [('time', 'Time'), ('amount', 'Amount')]:
        if feature_lower in df_recent.columns:
            recent_mean = float(df_recent[feature_lower].mean())
            ref_mean = ref_stats[feature_ref]['mean']
            ref_std = ref_stats[feature_ref]['std']
            
            # Drift is flagged if the recent mean shifts by more than 2 reference standard deviations
            deviation = abs(recent_mean - ref_mean)
            threshold = 2 * ref_std
            
            if deviation > threshold:
                drift_detected = True
                alerts.append({
                    "feature": feature_lower,
                    "type": "mean_shift",
                    "reference_mean": round(ref_mean, 2),
                    "recent_mean": round(recent_mean, 2),
                    "severity": "high"
                })
                
    # 4. Perform Check 2: Fraud Rate Shift
    reference_fraud_rate = 0.00173  # 0.173% in baseline dataset
    recent_fraud_count = int(df_recent['is_fraud'].sum())
    recent_fraud_rate = float(recent_fraud_count / sample_size)
    
    fraud_rate_flagged = False
    if recent_fraud_rate > (3 * reference_fraud_rate):
        drift_detected = True
        fraud_rate_flagged = True
        
    return {
        "drift_detected": drift_detected,
        "checked_at": checked_at,
        "sample_size": sample_size,
        "alerts": alerts,
        "fraud_rate": {
            "reference": reference_fraud_rate,
            "recent": round(recent_fraud_rate, 5),
            "flagged": fraud_rate_flagged
        }
    }


def get_prediction_stats(n_recent=100):
    """
    Computes summary prediction statistics from the predictions log file.
    Used for monitoring general model traffic and health.
    """
    log_path = 'predictions_log.jsonl'
    total_predictions = 0
    recent_transactions = []
    
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            for line in f:
                if line.strip():
                    total_predictions += 1
                    try:
                        recent_transactions.append(json.loads(line))
                    except Exception:
                        pass
                        
    if total_predictions == 0:
        return {
            "total_predictions": 0,
            "fraud_rate_recent": 0.0,
            "avg_fraud_probability_recent": 0.0,
            "last_prediction_timestamp": None,
            "recent_predictions": []
        }
        
    # Get last N recent predictions
    recent_subset = recent_transactions[-n_recent:]
    df_recent = pd.DataFrame(recent_subset)
    
    fraud_rate_recent = float(df_recent['is_fraud'].sum() / len(recent_subset))
    avg_prob_recent = float(df_recent['fraud_probability'].mean())
    last_timestamp = df_recent.iloc[-1]['timestamp']
    
    # Get last 10 predictions for the dashboard table (most recent first)
    recent_10 = recent_subset[-10:] if len(recent_subset) > 10 else recent_subset
    recent_10_reversed = list(reversed(recent_10))
    
    return {
        "total_predictions": total_predictions,
        "fraud_rate_recent": round(fraud_rate_recent, 5),
        "avg_fraud_probability_recent": round(avg_prob_recent, 5),
        "last_prediction_timestamp": last_timestamp,
        "recent_predictions": recent_10_reversed
    }
