import os
import csv
import time
import json
import pandas as pd
import joblib
from datetime import datetime, timezone
from investigation_agent import investigate_transaction
from investigation_context import get_mock_identity_for_id

# 1. Load API Key and configure environment
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as ef:
            for line in ef:
                line_stripped = line.strip()
                if line_stripped and not line_stripped.startswith("#"):
                    parts = line_stripped.split("=", 1)
                    if len(parts) == 2 and parts[0].strip() == "GROQ_API_KEY":
                        api_key = parts[1].strip().strip("'").strip('"')
                        os.environ["GROQ_API_KEY"] = api_key
                        break
if not api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment or a .env file.")

def score_transaction(row, model, scaler):
    """
    Computes model fraud probability for a raw transaction row.
    """
    scaling_df = pd.DataFrame([[float(row['Time']), float(row['Amount'])]], columns=['Time', 'Amount'])
    scaled_vals = scaler.transform(scaling_df)
    scaled_time = scaled_vals[0][0]
    scaled_amount = scaled_vals[0][1]
    
    features_dict = {
        'Time': scaled_time,
        'Amount': scaled_amount,
    }
    for i in range(1, 29):
        features_dict[f'V{i}'] = float(row[f'V{i}'])
        
    sample_df = pd.DataFrame([features_dict])
    feature_names = model.get_booster().feature_names
    sample_df = sample_df[feature_names]
    
    prob = float(model.predict_proba(sample_df)[0][1])
    return prob

def main():
    model_path = "model.pkl"
    scaler_path = "scaler.pkl"
    csv_path = "creditcard.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("model.pkl or scaler.pkl not found. Run train.py first.")
        
    print("Loading model artifacts...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print("Loading creditcard.csv (this might take a few seconds)...")
    df = pd.read_csv(csv_path)
    
    # Separate fraud and normal rows to sample from
    df_fraud = df[df["Class"] == 1]
    df_normal = df[df["Class"] == 0]
    
    low_candidates = []
    med_candidates = []
    high_candidates = []
    
    # 2. Select candidates dynamically
    print("Scanning fraud rows for high/medium candidates...")
    for idx, row in df_fraud.sample(frac=1.0, random_state=42).iterrows():
        prob = score_transaction(row, model, scaler)
        if prob > 0.70 and len(high_candidates) < 6:
            high_candidates.append((idx, row, prob))
        elif 0.20 <= prob <= 0.70 and len(med_candidates) < 6:
            med_candidates.append((idx, row, prob))
        if len(high_candidates) == 6 and len(med_candidates) == 6:
            break
            
    print("Scanning normal rows for low/medium candidates...")
    for idx, row in df_normal.sample(frac=1.0, random_state=42).iterrows():
        prob = score_transaction(row, model, scaler)
        if prob < 0.20 and len(low_candidates) < 6:
            low_candidates.append((idx, row, prob))
        elif 0.20 <= prob <= 0.70 and len(med_candidates) < 6:
            med_candidates.append((idx, row, prob))
        if len(low_candidates) == 6 and len(med_candidates) == 6:
            break
            
    candidates = low_candidates + med_candidates + high_candidates
    print(f"Selected {len(candidates)} total transactions for evaluation:")
    print(f"  - Low risk: {len(low_candidates)}")
    print(f"  - Medium risk: {len(med_candidates)}")
    print(f"  - High risk: {len(high_candidates)}")
    
    # 3. Register candidates in predictions_log.jsonl
    log_path = "predictions_log.jsonl"
    existing_ids = set()
    if os.path.exists(log_path):
        with open(log_path, 'r') as lf:
            for line in lf:
                if line.strip():
                    try:
                        existing_ids.add(json.loads(line).get("transaction_id"))
                    except Exception:
                        pass
                        
    mock_entries_to_append = []
    for idx, row, prob in candidates:
        tx_id = f"batch-test-{idx}"
        if tx_id in existing_ids:
            continue
        entry = {
            "transaction_id": tx_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "time": float(row["Time"]),
            "amount": float(row["Amount"]),
            "fraud_probability": prob,
            "is_fraud": bool(row["Class"] == 1),
            "risk_level": "high" if prob > 0.7 else ("medium" if prob >= 0.2 else "low")
        }
        for i in range(1, 29):
            entry[f"V{i}"] = float(row[f"V{i}"])
        mock_entries_to_append.append(entry)
        
    if mock_entries_to_append:
        with open(log_path, 'a') as lf:
            for entry in mock_entries_to_append:
                lf.write(json.dumps(entry) + "\n")
        print(f"Registered {len(mock_entries_to_append)} mock transactions in {log_path}.")
        
    # 4. Run investigation loops
    results = []
    for i, (idx, row, prob) in enumerate(candidates):
        tx_id = f"batch-test-{idx}"
        account_id, _ = get_mock_identity_for_id(tx_id)
        before_time = float(row["Time"])
        
        print(f"\nTesting transaction {i+1}/{len(candidates)}: ID={tx_id}, Risk Tier={'High' if prob > 0.7 else ('Medium' if prob >= 0.2 else 'Low')} (Score: {prob:.4f})...")
        
        decision = investigate_transaction(transaction_id=tx_id, account_id=account_id, before_time=before_time)
        
        if "error" in decision:
            print(f"  [ERROR] {decision['error']}")
            
        results.append({
            "transaction_index": idx,
            "account_id": account_id,
            "model_fraud_probability": prob,
            "agent_decision": decision.get("decision", "escalate"),
            "agent_confidence": decision.get("confidence", 0.5),
            "tools_used": ",".join(decision.get("tools_used", [])),
            "rationale_summary": decision.get("rationale", "")[:100].replace("\n", " ")
        })
        
        # Add basic rate-limit delay between calls
        time.sleep(2.5)
        
    # 5. Save results to CSV
    results_csv = "batch_test_results.csv"
    fieldnames = [
        "transaction_index",
        "account_id",
        "model_fraud_probability",
        "agent_decision",
        "agent_confidence",
        "tools_used",
        "rationale_summary"
    ]
    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print(f"\nSaved batch test results to '{results_csv}'.")
    
    # 6. Calculate statistics and print report
    total = len(results)
    decisions = {"block": 0, "escalate": 0, "dismiss": 0}
    agreed_count = 0
    disagreements = []
    
    for r in results:
        prob = r["model_fraud_probability"]
        dec = r["agent_decision"].lower()
        decisions[dec] = decisions.get(dec, 0) + 1
        
        # Agreement Logic:
        # - High risk (model probability > 70%): Agent decision is "block" or "escalate"
        # - Low risk (model probability < 20%): Agent decision is "dismiss"
        # - Medium risk (20-70%): Any other combination is considered disagreement under literal requirements
        agreed = False
        if prob > 0.70:
            if dec in ["block", "escalate"]:
                agreed = True
        elif prob < 0.20:
            if dec == "dismiss":
                agreed = True
        else:
            # Medium risk candidates do not fall into high or low bins, so they are marked as disagreement
            # as per the instruction: "anything else = disagreement"
            agreed = False
            
        if agreed:
            agreed_count += 1
        else:
            disagreements.append(r)
            
    agreement_rate = (agreed_count / total) * 100 if total > 0 else 0.0
    
    print("\n==================================================")
    print("BATCH TESTING SUMMARY REPORT")
    print("==================================================")
    print(f"Total transactions tested: {total}")
    print(f"Agent decision breakdown:")
    print(f"  - Block: {decisions.get('block', 0)}")
    print(f"  - Escalate: {decisions.get('escalate', 0)}")
    print(f"  - Dismiss: {decisions.get('dismiss', 0)}")
    print(f"Agreement Rate: {agreement_rate:.2f}%")
    print("==================================================")
    
    if disagreements:
        print("\nDisagreement Cases (Agent vs Model Risk Tier):")
        print("--------------------------------------------------")
        for d in disagreements:
            print(f"Index: {d['transaction_index']} | Model Prob: {d['model_fraud_probability']:.4f} | Agent Dec: {d['agent_decision']} | Conf: {d['agent_confidence']:.2f}")
            print(f"  Rationale Summary: {d['rationale_summary']}...")
            print("-" * 50)
    else:
        print("\nNo disagreement cases found.")

if __name__ == "__main__":
    main()
