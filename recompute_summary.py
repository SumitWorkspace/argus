import csv
import os

def is_valid_row(row):
    """
    Checks if a row in the batch results is a valid successful agent execution.
    Excludes empty, missing, or error/failed runs.
    """
    decision = row.get("agent_decision")
    if not decision or decision.strip() == "" or decision.lower() not in ["block", "escalate", "dismiss"]:
        return False
        
    tools = row.get("tools_used")
    rationale = row.get("rationale_summary")
    
    if not tools or tools.strip() == "":
        return False
    if not rationale or rationale.strip() == "":
        return False
        
    # Check if the rationale indicates an API failure or system error
    rat_lower = rationale.lower()
    if "api transaction request failed" in rat_lower or "failed to load" in rat_lower or "error" in rat_lower:
        return False
        
    return True

def main():
    csv_path = "batch_test_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found. Please ensure the batch test results file exists.")
        return

    valid_rows = []
    excluded_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if is_valid_row(row):
                # Parse numeric fields
                row["model_fraud_probability"] = float(row["model_fraud_probability"])
                row["agent_confidence"] = float(row["agent_confidence"])
                valid_rows.append(row)
            else:
                excluded_count += 1

    total_valid = len(valid_rows)
    decisions = {"block": 0, "escalate": 0, "dismiss": 0}
    agreed_count = 0
    disagreements = []

    for row in valid_rows:
        prob = row["model_fraud_probability"]
        dec = row["agent_decision"].lower()
        decisions[dec] = decisions.get(dec, 0) + 1

        # Corrected Agreement Logic:
        # - High risk (model > 0.7): agreement if block or escalate; disagreement if dismiss
        # - Low risk (model < 0.2): agreement if dismiss; disagreement if block or escalate
        # - Medium risk (0.2-0.7): agreement if escalate; disagreement if block or dismiss
        agreed = False
        if prob > 0.7:
            if dec in ["block", "escalate"]:
                agreed = True
        elif prob < 0.2:
            if dec == "dismiss":
                agreed = True
        else:
            # Medium risk (0.2 - 0.7)
            if dec == "escalate":
                agreed = True

        if agreed:
            agreed_count += 1
        else:
            disagreements.append(row)

    agreement_rate = (agreed_count / total_valid) * 100 if total_valid > 0 else 0.0

    print("\n==================================================")
    print("RECOMPUTED BATCH TESTING SUMMARY REPORT")
    print("==================================================")
    print(f"Total valid transactions counted: {total_valid}")
    print(f"Excluded failed/error transactions: {excluded_count}")
    print(f"Agent decision breakdown:")
    print(f"  - Block: {decisions.get('block', 0)}")
    print(f"  - Escalate: {decisions.get('escalate', 0)}")
    print(f"  - Dismiss: {decisions.get('dismiss', 0)}")
    print(f"Corrected Agreement Rate: {agreement_rate:.2f}%")
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
