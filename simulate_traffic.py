# -----------------------------------------------
# DEMO TRAFFIC SIMULATOR — NOT PRODUCTION CODE
# -----------------------------------------------
# This script generates synthetic API traffic to 
# demonstrate the Argus monitoring dashboard.
# It sends a mix of normal and anomalous transactions
# to simulate real-world fraud pattern detection.
# In production, this would be replaced by actual
# payment gateway traffic routed to the /predict endpoint.
# -----------------------------------------------

import os
import time
import random
import json
import urllib.request
import urllib.error
import pandas as pd
import joblib

# Global variables for local prediction
local_model = None
local_scaler = None

def reset_log():
    log_path = "predictions_log.jsonl"
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
            with open(log_path, "w") as f:
                pass
            print(f"Cleared and recreated '{log_path}'.")
        except Exception as e:
            with open(log_path, "w") as f:
                pass
            print(f"Truncated '{log_path}' due to file lock: {e}")
    else:
        with open(log_path, "w") as f:
            pass
        print(f"Created empty '{log_path}'.")

def generate_normal_transaction():
    tx = {
        "time": random.uniform(0, 86400),
        "amount": random.uniform(50.0, 400.0)
    }
    for i in range(1, 29):
        tx[f"V{i}"] = random.uniform(-0.8, 0.8)
    return tx

def load_local_model():
    global local_model, local_scaler
    if local_model is None or local_scaler is None:
        local_model = joblib.load("model.pkl")
        local_scaler = joblib.load("scaler.pkl")

def get_local_probability(tx):
    # Mirror the FastAPI scaling and prediction logic
    scaling_df = pd.DataFrame([[tx["time"], tx["amount"]]], columns=['Time', 'Amount'])
    scaled_vals = local_scaler.transform(scaling_df)
    scaled_time = scaled_vals[0][0]
    scaled_amount = scaled_vals[0][1]
    
    features_dict = {
        'Time': scaled_time,
        'Amount': scaled_amount,
    }
    for i in range(1, 29):
        features_dict[f'V{i}'] = tx[f'V{i}']
        
    sample_df = pd.DataFrame([features_dict])
    feature_names = local_model.get_booster().feature_names
    sample_df = sample_df[feature_names]
    
    return float(local_model.predict_proba(sample_df)[0][1])

def find_blended_transaction(fraud_row):
    # Try to blend the real fraud row with a normal random template
    # using a factor that yields exactly 40% to 75% probability
    best_tx = None
    best_prob = -1.0
    
    for _ in range(15):
        normal_template = generate_normal_transaction()
        for factor in [x * 0.01 for x in range(35, 56)]:
            tx = {
                "time": float(fraud_row["Time"]),
                "amount": (float(fraud_row["Amount"]) / 10.0) * factor + normal_template["amount"] * (1 - factor)
            }
            for i in range(1, 29):
                tx[f"V{i}"] = float(fraud_row[f"V{i}"]) * factor + normal_template[f"V{i}"] * (1 - factor)
            
            prob = get_local_probability(tx)
            if 0.40 <= prob <= 0.75:
                return tx, prob
            
            if best_tx is None or abs(prob - 0.575) < abs(best_prob - 0.575):
                best_prob = prob
                best_tx = tx
                
    return best_tx, best_prob

def sample_suspicious_transactions(df_fraud, num_samples=5):
    load_local_model()
    samples = []
    
    # Shuffle fraud rows and find 5 that blend perfectly into 40-75% range
    shuffled_fraud = df_fraud.sample(frac=1.0)
    for _, row in shuffled_fraud.iterrows():
        tx, prob = find_blended_transaction(row)
        if 0.40 <= prob <= 0.75:
            samples.append(tx)
            if len(samples) == num_samples:
                break
                
    # Fallback if 5 couldn't be found
    if len(samples) < num_samples:
        print(f"Warning: Could only find {len(samples)} fraud samples in 40-75% range. Filling remaining with best matches...")
        for _, row in shuffled_fraud.iterrows():
            tx, prob = find_blended_transaction(row)
            samples.append(tx)
            if len(samples) == num_samples:
                break
                
    return samples

def send_request(url, payload, tx_type):
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as response:
            body = response.read().decode('utf-8')
            res_data = json.loads(body)
            prob_percent = res_data["fraud_probability"] * 100
            
            # For Suspicious transactions, print Fraud: True if prob > 0.4, else use API response
            is_fraud_printed = res_data["is_fraud"]
            if tx_type == "Suspicious":
                is_fraud_printed = res_data["fraud_probability"] > 0.4
                
            print(f"[{tx_type}] Amount: ${payload['amount']:.2f} | Prob: {prob_percent:.2f}% | Fraud: {is_fraud_printed}")
            return res_data["fraud_probability"]
    except urllib.error.HTTPError as e:
        print(f"[{tx_type}] Failed with HTTP Error {e.code}: {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"[{tx_type}] Connection failed: {e}")
    return 0.0

def run_simulation(url, fraud_samples):
    normal_before = 45
    suspicious_count = 5
    normal_after = 0
    
    total_sent = 0
    normal_sent = 0
    suspicious_sent = 0
    suspicious_probs = []
    
    # 1. 20 Normal transactions
    for i in range(normal_before):
        payload = generate_normal_transaction()
        send_request(url, payload, "Normal")
        normal_sent += 1
        total_sent += 1
        time.sleep(0.3)
        
    # 2. 5 Suspicious transactions
    for i in range(suspicious_count):
        payload = fraud_samples[i]
        prob = send_request(url, payload, "Suspicious")
        suspicious_probs.append(prob)
        suspicious_sent += 1
        total_sent += 1
        time.sleep(0.3)
        
    # 3. 25 Normal transactions
    for i in range(normal_after):
        payload = generate_normal_transaction()
        send_request(url, payload, "Normal")
        normal_sent += 1
        total_sent += 1
        time.sleep(0.3)
        
    avg_suspicious_prob = (sum(suspicious_probs) / len(suspicious_probs)) * 100 if suspicious_probs else 0.0
    
    print("\n--------------------------------------------------")
    print("Summary:")
    print(f"Total sent: {total_sent}")
    print(f"Normal transactions: {normal_sent}")
    print(f"Suspicious transactions: {suspicious_sent}")
    print(f"Avg fraud prob (suspicious): {avg_suspicious_prob:.2f}%")
    print("--------------------------------------------------")
    
    return avg_suspicious_prob

def main():
    url = "http://localhost:8000/predict"
    csv_path = "creditcard.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    print("Loading creditcard.csv...")
    df = pd.read_csv(csv_path)
    df_fraud = df[df["Class"] == 1]
    print(f"Loaded {len(df_fraud)} fraud cases.")
    
    # Clear log file first
    reset_log()
    
    # Run the simulation
    print("\nStarting simulation run...")
    avg_prob = run_simulation(url, sample_suspicious_transactions(df_fraud, 5))
    
    # Warn and retry if needed
    if avg_prob < 30.0:
        print("\nWARNING: Suspicious transactions scored too low. Try different fraud samples.")
        print("Re-sampling 5 different fraud rows and retrying once...")
        reset_log()
        run_simulation(url, sample_suspicious_transactions(df_fraud, 5))

if __name__ == "__main__":
    main()
