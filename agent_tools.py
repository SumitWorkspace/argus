import os
import json
import pandas as pd
import numpy as np
import joblib

# Import functions from investigation_context
from investigation_context import (
    get_account_history,
    get_pattern_deviation,
    add_mock_identity_columns
)

# Global cache for data and model artifacts
_transactions_df = None
_model = None
_scaler = None

def _load_data():
    """
    Helper to lazily load and cache the transaction dataset.
    Augments the dataset with mock identity columns for evaluation.
    """
    global _transactions_df
    if _transactions_df is None:
        csv_path = "creditcard.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file '{csv_path}' not found in the project root.")
        print(f"Loading transaction dataset from '{csv_path}'...")
        df = pd.read_csv(csv_path)
        # Augment with mock identity columns for test / simulation capability
        _transactions_df = add_mock_identity_columns(df)
    return _transactions_df

def _load_model_artifacts():
    """
    Helper to lazily load and cache model and scaler artifacts.
    """
    global _model, _scaler
    if _model is None or _scaler is None:
        model_path = "model.pkl"
        scaler_path = "scaler.pkl"
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model (model.pkl) or Scaler (scaler.pkl) artifact not found. Please run train.py first.")
        print(f"Loading ML artifacts '{model_path}' and '{scaler_path}'...")
        _model = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)
    return _model, _scaler

TOOL_DEFINITIONS = [
    {
        "name": "tool_get_account_history",
        "description": "Retrieves the most recent historical transactions for a given account before a specific timestamp. Use this tool to examine the customer's typical purchasing behavior and identify pattern changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "The unique identifier of the bank account (e.g., 'ACC_039')."
                },
                "before_time": {
                    "type": "number",
                    "description": "The cutoff time in seconds elapsed since the dataset start. Only transactions strictly before this time are returned to avoid data leakage."
                }
            },
            "required": ["account_id", "before_time"]
        }
    },
    {
        "name": "tool_get_pattern_deviation",
        "description": "Calculates statistical and lifestyle deviations of a specific transaction against the historical baseline of the account. Returns time since the account's previous transaction in minutes (time_since_last_transaction), transaction velocity in the last 60 minutes, amount z-score, and new merchant category flags.",
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "The unique identifier of the bank account (e.g., 'ACC_039')."
                },
                "transaction_id": {
                    "type": "string",
                    "description": "The unique UUID identifier of the target transaction being investigated."
                },
                "before_time": {
                    "type": "number",
                    "description": "The timestamp of the target transaction in seconds. Used as the historical cutoff to prevent data leakage."
                }
            },
            "required": ["account_id", "transaction_id", "before_time"]
        }
    },
    {
        "name": "tool_get_model_score",
        "description": "Scores a specific transaction using the trained XGBoost model. Returns the predictive probability of fraud.",
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The unique UUID identifier of the target transaction."
                }
            },
            "required": ["transaction_id"]
        }
    }
]

def _get_logged_transaction(transaction_id):
    """
    Helper to look up a transaction by ID in predictions_log.jsonl.
    """
    log_path = "predictions_log.jsonl"
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file '{log_path}' not found. Cannot retrieve transaction '{transaction_id}'.")
    with open(log_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if data.get("transaction_id") == transaction_id:
                        return data
                except Exception:
                    pass
    raise ValueError(f"Transaction ID '{transaction_id}' not found in prediction logs.")

def tool_get_account_history(account_id, before_time):
    """
    Tool function to get account transaction history.

    Parameters:
    -----------
    account_id : str
        The account identifier.
    before_time : float
        Filter cutoff to get only transactions before this time.

    Returns:
    --------
    list
        List of dicts representing the historical transaction records (containing only Time, Amount, merchant_category, and Class).
    """
    df = _load_data()
    # Call core history function (limit default to 50)
    history_df = get_account_history(account_id, df, limit=50, before_time=before_time)
    
    # Filter only columns of interest for LLM reasoning (drop V1-V28 PCA features)
    keep_cols = [c for c in ['Time', 'Amount', 'merchant_category', 'Class'] if c in history_df.columns]
    history_filtered = history_df[keep_cols]
    
    # Convert DataFrame to JSON-serializable list of dicts
    records = history_filtered.to_dict(orient='records')
    return records

def tool_get_pattern_deviation(account_id, transaction_id, before_time):
    """
    Tool function to calculate behavioral deviation statistics.

    Parameters:
    -----------
    account_id : str
        The account identifier.
    transaction_id : str
        The unique ID of the transaction to evaluate.
    before_time : float
        Filter cutoff timestamp.

    Returns:
    --------
    dict
        Dictionary of deviation metrics.
    """
    df = _load_data()
    tx_logged = _get_logged_transaction(transaction_id)
    
    # Get mock merchant_category deterministically for this transaction_id
    from investigation_context import get_mock_identity_for_id
    _, merchant_category = get_mock_identity_for_id(transaction_id)
    
    # Construct a Series/dict representing the target transaction
    target_tx = pd.Series({
        'Time': float(tx_logged['time']),
        'Amount': float(tx_logged['amount']),
        'merchant_category': merchant_category
    })
    
    # Pass target transaction and historical dataframe to core logic
    deviation = get_pattern_deviation(account_id, target_tx, df)
    return deviation

def tool_get_model_score(transaction_id):
    """
    Tool function to score a transaction using the trained XGBoost model.

    Parameters:
    -----------
    transaction_id : str
        The unique ID of the transaction to evaluate.

    Returns:
    --------
    dict
        Dictionary containing prediction results.
    """
    tx_logged = _get_logged_transaction(transaction_id)
    model, scaler = _load_model_artifacts()
    
    # 1. Scale numerical features (Time and Amount)
    scaling_df = pd.DataFrame([[float(tx_logged['time']), float(tx_logged['amount'])]], columns=['Time', 'Amount'])
    scaled_vals = scaler.transform(scaling_df)
    scaled_time = scaled_vals[0][0]
    scaled_amount = scaled_vals[0][1]
    
    # 2. Assemble features in the order expected by XGBoost booster
    features_dict = {
        'Time': scaled_time,
        'Amount': scaled_amount,
    }
    for i in range(1, 29):
        # Retrieve V1-V28 from the logged prediction entry
        features_dict[f'V{i}'] = float(tx_logged[f'V{i}'])
        
    sample_df = pd.DataFrame([features_dict])
    feature_names = model.get_booster().feature_names
    sample_df = sample_df[feature_names]
    
    # 3. Perform inference
    fraud_probability = float(model.predict_proba(sample_df)[0][1])
    is_fraud_pred = bool(model.predict(sample_df)[0] == 1)
    
    return {
        "transaction_id": transaction_id,
        "fraud_probability": fraud_probability,
        "predicted_class": int(is_fraud_pred)
    }

if __name__ == "__main__":
    # Test harness execution using index 541 (a known Class=1 fraud transaction)
    sample_idx = 541
    print("==================================================")
    
    # 1. Load raw transaction from dataset to prepare a log entry mock
    df = _load_data()
    tx_raw = df.iloc[sample_idx]
    
    # Deterministic mock identities
    from investigation_context import get_mock_identity_for_id
    sample_id = "541-uuid-placeholder-for-test"
    account_id, _ = get_mock_identity_for_id(sample_id)
    before_time = float(tx_raw["Time"])
    
    print(f"Target transaction at index {sample_idx} mapped to UUID {sample_id}:")
    print(f"  Account ID: {account_id}")
    print(f"  Time: {before_time}")
    print(f"  Amount: ${tx_raw['Amount']:.2f}")
    
    # Build mock log entry matching the new predictions_log.jsonl format with V1-V28
    log_entry = {
        "transaction_id": sample_id,
        "timestamp": "2026-07-03T04:00:00.000000+00:00",
        "time": before_time,
        "amount": float(tx_raw["Amount"]),
        "fraud_probability": 0.9982,
        "is_fraud": True,
        "risk_level": "high"
    }
    for i in range(1, 29):
        log_entry[f"V{i}"] = float(tx_raw[f"V{i}"])
        
    # Append the test transaction log entry if not exists
    log_path = "predictions_log.jsonl"
    existing_ids = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as lf:
            for line in lf:
                if line.strip():
                    try:
                        existing_ids.append(json.loads(line).get("transaction_id"))
                    except Exception:
                        pass
    if sample_id not in existing_ids:
        with open(log_path, "a") as lf:
            lf.write(json.dumps(log_entry) + "\n")
            
    print("==================================================\n")

    # 1. Test Model Scoring Tool
    print("Testing tool_get_model_score...")
    score_res = tool_get_model_score(sample_id)
    print(json.dumps(score_res, indent=2))
    print()

    # 2. Test Pattern Deviation Tool
    print("Testing tool_get_pattern_deviation...")
    deviation_res = tool_get_pattern_deviation(account_id, sample_id, before_time)
    print(json.dumps(deviation_res, indent=2))
    print()

    # 3. Test Account History Tool (Show first 3 records for brevity)
    print("Testing tool_get_account_history...")
    history_res = tool_get_account_history(account_id, before_time)
    print(f"Total history entries retrieved: {len(history_res)}")
    print("First 3 historical records:")
    print(json.dumps(history_res[:3], indent=2))
