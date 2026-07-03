# Run with: uvicorn app:app --reload --port 8000
# Test at: http://localhost:8000/docs

import os
import uuid
import json
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Global model and scaler variables
model = None
scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    model_path = "model.pkl"
    scaler_path = "scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise RuntimeError("Serialized artifacts (model.pkl, scaler.pkl) not found. Please run train.py first.")
        
    print(f"Loading model from '{model_path}'...")
    model = joblib.load(model_path)
    
    print(f"Loading scaler from '{scaler_path}'...")
    scaler = joblib.load(scaler_path)
    
    # Compute or load baseline reference statistics at startup
    try:
        from drift_detector import compute_reference_stats
        compute_reference_stats()
    except Exception as e:
        print(f"Warning: Failed to compute/load reference stats: {e}")
        
    yield
    # Clean up resources on shutdown if needed

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for predicting credit card transaction fraud using XGBoost",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TransactionRequest(BaseModel):
    time: float = Field(..., description="Seconds elapsed since first transaction")
    amount: float = Field(..., description="Transaction amount")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    class Config:
        json_schema_extra = {
            "example": {
                "time": 45000.0,
                "amount": 250.75,
                "V1": -1.35, "V2": 0.21, "V3": 2.53, "V4": 1.37,
                "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09,
                "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61,
                "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47,
                "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
                "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06,
                "V25": 0.12, "V26": -0.18, "V27": 0.13, "V28": -0.02
            }
        }

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    transaction_id: str
    timestamp: str

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None and scaler is not None,
        "model_type": "XGBoost",
        "features_expected": 30
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TransactionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model and/or scaler not loaded on server.")
        
    try:
        # 1. Scale 'time' and 'amount' using the loaded scaler
        # Pass a DataFrame with columns ['Time', 'Amount'] to avoid UserWarning
        scaling_df = pd.DataFrame([[request.time, request.amount]], columns=['Time', 'Amount'])
        scaled_vals = scaler.transform(scaling_df)
        scaled_time = scaled_vals[0][0]
        scaled_amount = scaled_vals[0][1]
        
        # 2. Assemble features in the exact training order:
        # Time, Amount, V1-V28
        features_dict = {
            'Time': scaled_time,
            'Amount': scaled_amount,
        }
        for i in range(1, 29):
            features_dict[f'V{i}'] = getattr(request, f'V{i}')
            
        sample_df = pd.DataFrame([features_dict])
        
        # Explicit column alignment using model's expected features
        feature_names = model.get_booster().feature_names
        sample_df = sample_df[feature_names]
        
        # 3. Predict probability and class
        is_fraud_pred = bool(model.predict(sample_df)[0] == 1)
        fraud_probability = float(model.predict_proba(sample_df)[0][1])
        
        # Determine risk level
        if fraud_probability > 0.7:
            risk_level = "high"
        elif fraud_probability > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
            
        transaction_id = str(uuid.uuid4())
        timestamp_str = datetime.now(timezone.utc).isoformat()
        
        # 4. Log prediction to predictions_log.jsonl
        log_entry = {
            "transaction_id": transaction_id,
            "timestamp": timestamp_str,
            "time": request.time,
            "amount": request.amount,
            "fraud_probability": fraud_probability,
            "is_fraud": is_fraud_pred,
            "risk_level": risk_level
        }
        
        with open("predictions_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        return PredictionResponse(
            is_fraud=is_fraud_pred,
            fraud_probability=fraud_probability,
            risk_level=risk_level,
            transaction_id=transaction_id,
            timestamp=timestamp_str
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/drift")
async def drift(n: int = 100):
    """
    Returns a data drift detection report comparing recent logged transaction
    features and predictions against baseline training distribution parameters.
    """
    from drift_detector import detect_drift
    return detect_drift(n_recent=n)


@app.get("/stats")
async def stats(n: int = 100):
    """
    Returns general performance and volume metrics calculated from the logged transactions.
    """
    from drift_detector import get_prediction_stats
    return get_prediction_stats(n_recent=n)


@app.get("/dashboard")
async def get_dashboard():
    """
    Serves the live operations monitoring dashboard.
    """
    from fastapi.responses import FileResponse
    return FileResponse("dashboard.html")


@app.post("/investigate/{transaction_index}")
async def investigate_endpoint(transaction_index: int):
    """
    Runs the ReAct fraud investigation agent on a specified transaction.
    """
    try:
        from agent_tools import _load_data
        from investigation_agent import investigate_transaction
        
        df = _load_data()
        if transaction_index < 0 or transaction_index >= len(df):
            raise HTTPException(status_code=404, detail=f"Transaction index {transaction_index} not found in the dataset.")
            
        target_tx = df.iloc[transaction_index]
        account_id = str(target_tx['account_id'])
        before_time = float(target_tx['Time'])
        
        result = investigate_transaction(transaction_index, account_id, before_time)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investigation agent error: {str(e)}")

