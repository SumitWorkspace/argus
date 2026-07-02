# Expected feature order:
# ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb
import joblib

def load_data():
    """
    Loads the credit card fraud detection dataset from a local CSV file.
    """
    csv_path = 'creditcard.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file '{csv_path}' not found in the project root directory.")
    print(f"Loading dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    return df

def verify_saved_artifacts(scale_cols):
    """
    Simulates a single-sample inference request (like a FastAPI request)
    by loading model.pkl and scaler.pkl and making a prediction.
    """
    print("\n--- Verifying Saved Artifacts (Model & Scaler Verification Test) ---")
    
    # 1. Load artifacts
    model_filename = 'model.pkl'
    scaler_filename = 'scaler.pkl'
    
    if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
        print("Error: Serialized artifacts not found. Cannot run verification.")
        return
        
    loaded_model = joblib.load(model_filename)
    loaded_scaler = joblib.load(scaler_filename)
    print(f"Loaded '{model_filename}' and '{scaler_filename}' successfully.")
    
    # 2. Create a mock single-sample request
    print("Creating a mock transaction request...")
    raw_sample = {
        'Time': 86400.0,
        'Amount': 750.0,
    }
    # Fill V1-V28 PCA features with 0.0
    for i in range(1, 29):
        raw_sample[f'V{i}'] = 0.0
        
    sample_df = pd.DataFrame([raw_sample])
    
    # Scale Time and Amount
    sample_df[scale_cols] = loaded_scaler.transform(sample_df[scale_cols])
    
    # Reorder columns to match training features expected by XGBoost
    feature_names = loaded_model.get_booster().feature_names
    sample_df = sample_df[feature_names]
    
    print("\nPreprocessed Mock Sample:")
    print(sample_df)
    
    # 3. Predict
    prediction = loaded_model.predict(sample_df)[0]
    prediction_proba = loaded_model.predict_proba(sample_df)[0][1]
    
    print("\nInference Output:")
    print(f"  Prediction Class: {prediction} ({'Fraud' if prediction == 1 else 'Non-Fraud'})")
    print(f"  Fraud Probability: {prediction_proba:.4f}")
    print("Verification completed successfully!")

def main():
    print("==================================================")
    print("Starting Fraud Detection Model Training Pipeline")
    print("==================================================")
    
    # --------------------------------------------------
    # Step 1: Load Dataset
    # --------------------------------------------------
    df = load_data()
    print(f"\nDataset loaded. Shape: {df.shape}")
    
    # --------------------------------------------------
    # Step 2: Exploratory Data Analysis (EDA)
    # --------------------------------------------------
    print("\n--- Basic EDA ---")
    print("Columns:", list(df.columns))
    
    target_col = 'Class'
    scale_cols = ['Time', 'Amount']
    
    # Display Class Imbalance
    class_counts = df[target_col].value_counts()
    class_percentages = df[target_col].value_counts(normalize=True) * 100
    print("\nClass Imbalance:")
    for cls in class_counts.index:
        label = "Fraud" if cls == 1 else "Non-Fraud"
        print(f"  {label} ({cls}): {class_counts[cls]} transactions ({class_percentages[cls]:.3f}%)")
        
    # Basic statistics
    print("\nBasic Dataset Statistics:")
    print(df[scale_cols].describe().round(2))
    
    # --------------------------------------------------
    # Step 3: Preprocessing
    # --------------------------------------------------
    print("\n--- Preprocessing ---")
    
    # Split into features (X) and target (y) with explicit feature order
    feature_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    X = df[feature_cols]
    y = df[target_col]
    
    # Train/test split (80/20) with stratification to preserve imbalance ratio
    print("Splitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]} samples")
    print(f"Test size: {X_test.shape[0]} samples")
    
    # Scale numerical features (Standard Scaling for Time and Amount)
    print(f"Standard scaling numerical features: {scale_cols}")
    scaler = StandardScaler()
    
    # Fit on training data and transform both train and test
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])
    
    # --------------------------------------------------
    # Step 4: Model Training (XGBoost Classifier)
    # --------------------------------------------------
    print("\n--- Model Training ---")
    
    # Calculate class weight ratio for handling class imbalance (scale_pos_weight)
    # scale_pos_weight = count(negative) / count(positive)
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    print(f"Calculated scale_pos_weight for handling class imbalance: {scale_pos_weight:.2f}")
    
    # Initialize XGBoost classifier with simple hyperparameters
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train_scaled, y_train)
    print("Model training completed.")
    
    # --------------------------------------------------
    # Step 5: Model Evaluation
    # --------------------------------------------------
    print("\n--- Model Evaluation ---")
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print metrics
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud']))
    
    # --------------------------------------------------
    # Step 6: Save Model and Scaler Artifacts
    # --------------------------------------------------
    print("\n--- Saving Artifacts ---")
    model_filename = 'model.pkl'
    scaler_filename = 'scaler.pkl'
    
    joblib.dump(model, model_filename)
    print(f"Saved trained XGBoost model to '{model_filename}'")
    
    joblib.dump(scaler, scaler_filename)
    print(f"Saved fitted StandardScaler to '{scaler_filename}'")
    
    # --------------------------------------------------
    # Step 7: Verify Saved Artifacts
    # --------------------------------------------------
    verify_saved_artifacts(scale_cols)
    
    print("\nPipeline execution finished successfully.")

if __name__ == '__main__':
    main()
