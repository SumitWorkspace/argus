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
from typing import Optional

class FeedbackRequest(BaseModel):
    agent_decision: str
    agent_confidence: float
    human_verdict: str
    notes: Optional[str] = None

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
        for i in range(1, 29):
            log_entry[f"V{i}"] = float(getattr(request, f"V{i}"))
            
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
    from agent_tools import _load_data
    
    stats_data = get_prediction_stats(n_recent=n)
    
    # Map each recent prediction to its closest transaction index in the original dataset based on timestamp Time
    try:
        df = _load_data()
        for tx in stats_data.get("recent_predictions", []):
            tx_time = tx.get("time")
            if tx_time is not None:
                # Find closest index matching Time value
                closest_idx = int((df['Time'] - tx_time).abs().idxmin())
                tx["transaction_index"] = closest_idx
    except Exception as e:
        print(f"Warning: Failed to map transaction indices: {e}")
        
    return stats_data


@app.get("/dashboard")
async def get_dashboard():
    """
    Serves the live operations monitoring dashboard.
    """
    from fastapi.responses import FileResponse
    return FileResponse("dashboard.html")


@app.post("/investigate/{transaction_id}")
async def investigate_endpoint(transaction_id: str):
    """
    Runs the ReAct fraud investigation agent on a logged transaction.
    """
    try:
        from investigation_agent import investigate_transaction
        from investigation_context import get_mock_identity_for_id
        import json
        import os
        
        # Load transaction details directly from predictions_log.jsonl
        log_path = "predictions_log.jsonl"
        target_tx = None
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if data.get("transaction_id") == transaction_id:
                                target_tx = data
                                break
                        except Exception:
                            pass
                            
        if not target_tx:
            raise HTTPException(status_code=404, detail=f"Transaction ID {transaction_id} not found in the prediction logs.")
            
        account_id, _ = get_mock_identity_for_id(transaction_id)
        before_time = float(target_tx['time'])
        
        result = investigate_transaction(transaction_id, account_id, before_time)
        if "error" in result:
            if result.get("error") == "rate_limit":
                raise HTTPException(status_code=429, detail={"error": "rate_limit", "message": result.get("message")})
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investigation agent error: {str(e)}")


@app.post("/feedback/{transaction_id}")
async def post_feedback(transaction_id: str, request: FeedbackRequest):
    """
    Logs human feedback on the agent's risk decision for a transaction.
    """
    try:
        feedback_entry = {
            "transaction_id": transaction_id,
            "agent_decision": request.agent_decision,
            "agent_confidence": request.agent_confidence,
            "human_verdict": request.human_verdict,
            "notes": request.notes,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Append to agent_feedback.jsonl
        feedback_path = "agent_feedback.jsonl"
        with open(feedback_path, "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")
            
        return {"status": "success", "message": "Feedback recorded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@app.get("/agent-accuracy")
async def get_agent_accuracy():
    """
    Reads the human feedback logs and computes aggregate accuracy metrics,
    including overall accuracy, decision breakdown, and rolling trends.
    """
    feedback_path = "agent_feedback.jsonl"
    
    # Initialize empty response structure
    default_response = {
        "total_feedback_count": 0,
        "accuracy_rate": 0.0,
        "breakdown": {
            "block": {"correct": 0, "incorrect": 0},
            "escalate": {"correct": 0, "incorrect": 0},
            "dismiss": {"correct": 0, "incorrect": 0}
        },
        "rolling_accuracy": {
            "last_10": 0.0,
            "last_25": 0.0,
            "all": 0.0
        }
    }
    
    if not os.path.exists(feedback_path):
        return default_response
        
    entries = []
    try:
        with open(feedback_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read feedback logs: {str(e)}")
        
    if not entries:
        return default_response
        
    total_feedback_count = len(entries)
    
    # Calculate overall correctness
    correct_count = sum(1 for e in entries if e.get("human_verdict") == "correct")
    accuracy_rate = correct_count / total_feedback_count
    
    # Decision breakdown
    breakdown = {
        "block": {"correct": 0, "incorrect": 0},
        "escalate": {"correct": 0, "incorrect": 0},
        "dismiss": {"correct": 0, "incorrect": 0}
    }
    for e in entries:
        dec = e.get("agent_decision", "").lower()
        verd = e.get("human_verdict", "").lower()
        if dec in breakdown and verd in ["correct", "incorrect"]:
            breakdown[dec][verd] += 1
            
    # Rolling accuracy trends (last 10, last 25, and all feedback entries)
    def calc_accuracy(subset):
        if not subset:
            return 0.0
        corr = sum(1 for e in subset if e.get("human_verdict") == "correct")
        return corr / len(subset)
        
    rolling_accuracy = {
        "last_10": calc_accuracy(entries[-10:]),
        "last_25": calc_accuracy(entries[-25:]),
        "all": accuracy_rate
    }
    
    return {
        "total_feedback_count": total_feedback_count,
        "accuracy_rate": accuracy_rate,
        "breakdown": breakdown,
        "rolling_accuracy": rolling_accuracy
    }


