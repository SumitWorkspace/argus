import pandas as pd
import numpy as np

def _get_column_name(df, possible_names, default):
    """
    Finds the actual column name in the DataFrame from a list of possible names.
    Returns the first matching column name, or default if none are found.
    """
    for name in possible_names:
        if name in df.columns:
            return name
    return default

def _get_field(obj, keys, default=None):
    """
    Extracts a value from a dictionary, pandas Series, or object using a list of potential keys.
    """
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k]
    elif hasattr(obj, 'get'):
        for k in keys:
            val = obj.get(k)
            if val is not None:
                return val
    else:
        for k in keys:
            try:
                return obj[k]
            except (KeyError, IndexError, TypeError):
                pass
            try:
                return getattr(obj, k)
            except AttributeError:
                pass
    return default

def get_account_history(account_id, transactions_df, limit=50, before_time=None):
    """
    Retrieves the most recent N transactions for a given account from the transaction dataset,
    optionally filtering to only include transactions that occurred before a specific timestamp,
    sorted by time descending.

    Parameters:
    -----------
    account_id : str or int
        The identifier of the account to query.
    transactions_df : pandas.DataFrame
        The DataFrame containing transaction records.
    limit : int, default 50
        The maximum number of transactions to return.
    before_time : float, optional
        A cutoff timestamp (in seconds). If provided, only transactions occurring before
        this time are returned.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the account's historical transactions, sorted by Time descending.

    Why this matters for fraud investigation:
    -----------------------------------------
    Reviewing a user's recent transaction history allows investigators (and downstream agents)
    to understand their baseline purchasing habits. It helps differentiate between normal,
    frequent behaviors (such as regular grocery trips or routine bills) and sudden bursts of
    unusual activity, which are common in account takeover scenarios. Limiting history to prior
    transactions prevents data leakage during target transaction investigations.
    """
    account_col = _get_column_name(transactions_df, ['account_id', 'Account_ID', 'AccountID', 'accountId'], 'account_id')
    time_col = _get_column_name(transactions_df, ['Time', 'time'], 'Time')

    if account_col not in transactions_df.columns:
        raise ValueError(f"Account column '{account_col}' not found in transactions DataFrame.")

    # Filter for the account's transactions
    account_txs = transactions_df[transactions_df[account_col] == account_id]

    if before_time is not None:
        account_txs = account_txs[account_txs[time_col] < before_time]

    # Sort by time descending and limit
    sorted_txs = account_txs.sort_values(by=time_col, ascending=False)
    return sorted_txs.head(limit)

def get_pattern_deviation(account_id, transaction, transactions_df):
    """
    Compares a target transaction against the historical pattern of the specified account.
    Calculates statistical and behavioral deviations to contextualize fraud risk.

    Parameters:
    -----------
    account_id : str or int
        The account identifier.
    transaction : dict or pandas.Series
        The current transaction details being evaluated.
    transactions_df : pandas.DataFrame
        The historical transaction database.

    Returns:
    --------
    dict
        A dictionary containing:
        - time_since_last_transaction (float or None): Time elapsed since the previous transaction (in minutes).
        - transaction_velocity_1hr (int): Number of transactions in the last 60 minutes.
        - amount_zscore (float): Number of standard deviations the current amount deviates from the historical mean.
        - is_new_merchant_category (bool): True if the merchant category has not been seen in the history.

    Why this matters for fraud investigation:
    -----------------------------------------
    - time_since_last_transaction: Extremely short intervals between transactions (velocity spikes)
      often indicate automated attacks or rapid fraud card usage before the card is blocked.
    - transaction_velocity_1hr: Measuring the count of transactions in the last hour helps identify
      suspicious card-testing activity.
    - amount_zscore: Outsized transactions that deviate significantly from a user's normal spending range
      are a strong signal of unauthorized usage.
    - is_new_merchant_category: A transaction at a merchant category never visited before indicates a
      deviation in lifestyle patterns, a typical characteristic of stolen cards.
    """
    account_col = _get_column_name(transactions_df, ['account_id', 'Account_ID', 'AccountID', 'accountId'], 'account_id')
    time_col = _get_column_name(transactions_df, ['Time', 'time'], 'Time')
    amount_col = _get_column_name(transactions_df, ['Amount', 'amount'], 'Amount')
    merchant_col = _get_column_name(transactions_df, ['merchant_category', 'Merchant_Category', 'merchant', 'Merchant'], 'merchant_category')

    # Extract target transaction values
    curr_time = _get_field(transaction, ['Time', 'time'])
    curr_amount = _get_field(transaction, ['Amount', 'amount'])
    curr_merchant = _get_field(transaction, ['merchant_category', 'Merchant_Category', 'merchant', 'Merchant'])

    if curr_time is None or curr_amount is None:
        raise ValueError("Transaction must contain 'Time' (or 'time') and 'Amount' (or 'amount') fields.")

    # Filter historical transactions for this account prior to the current transaction
    history_df = transactions_df[
        (transactions_df[account_col] == account_id) &
        (transactions_df[time_col] < curr_time)
    ]

    # Sort history by time descending to find the most recent prior transactions
    history_sorted = history_df.sort_values(by=time_col, ascending=False)

    # 1. Time since last transaction (in minutes)
    time_since_last = None
    if not history_sorted.empty:
        last_time = history_sorted.iloc[0][time_col]
        time_since_last = float((curr_time - last_time) / 60.0)

    # 2. Transaction velocity (1 hour = 3600 seconds)
    velocity_1hr = int(
        ((history_df[time_col] >= curr_time - 3600) & (history_df[time_col] < curr_time)).sum()
    )

    # 3. Amount z-score
    amount_zscore = 0.0
    if not history_df.empty:
        amounts = history_df[amount_col]
        mean_amount = amounts.mean()
        std_amount = amounts.std(ddof=1)
        if not pd.isna(std_amount) and std_amount > 0.0:
            amount_zscore = float((curr_amount - mean_amount) / std_amount)

    # 4. New merchant category
    is_new_merchant = True
    if not history_df.empty and merchant_col in history_df.columns:
        seen_merchants = history_df[merchant_col].unique()
        if curr_merchant is not None:
            is_new_merchant = bool(curr_merchant not in seen_merchants)

    return {
        "time_since_last_transaction": time_since_last,
        "transaction_velocity_1hr": velocity_1hr,
        "amount_zscore": amount_zscore,
        "is_new_merchant_category": is_new_merchant
    }

def add_mock_identity_columns(df):
    """
    [MOCK DATA SCAFFOLDING]
    Adds synthetic 'account_id' and 'merchant_category' columns to the transaction DataFrame
    to simulate production systems for testing.
    """
    print("Injecting mock 'account_id' and 'merchant_category' columns for verification...")
    np.random.seed(42)
    
    # Define 50 synthetic accounts
    num_accounts = 50
    accounts = [f"ACC_{i:03d}" for i in range(num_accounts)]
    df['account_id'] = np.random.choice(accounts, size=len(df))

    # Define 10 merchant categories
    merchant_categories = [
        'groceries', 'gas_station', 'dining', 'online_retail', 'travel',
        'entertainment', 'electronics', 'apparel', 'services', 'other'
    ]

    # Assign preferred categories to each account to model realistic usage patterns
    account_pref = {}
    for acc in accounts:
        pref = np.random.choice(merchant_categories, size=3, replace=False)
        account_pref[acc] = pref

    # Assign category based on preferred categories (85% probability) or other categories (15%)
    categories_assigned = []
    for acc in df['account_id']:
        pref = account_pref[acc]
        non_pref = [c for c in merchant_categories if c not in pref]
        if np.random.rand() < 0.85:
            cat = np.random.choice(pref)
        else:
            cat = np.random.choice(non_pref)
        categories_assigned.append(cat)

    df['merchant_category'] = categories_assigned
    return df


def get_mock_identity_for_id(transaction_id):
    """
    [MOCK DATA SCAFFOLDING]
    Consistently maps a string or UUID transaction_id to a mock account_id and merchant_category
    using deterministic hashing. This ensures the same transaction always evaluates to the
    same identity across tools and endpoints.
    """
    import hashlib
    h = hashlib.sha256(str(transaction_id).encode('utf-8')).hexdigest()
    val = int(h, 16)
    
    # 50 accounts: ACC_000 to ACC_049
    num_accounts = 50
    accounts = [f"ACC_{i:03d}" for i in range(num_accounts)]
    acc_idx = val % num_accounts
    account_id = accounts[acc_idx]
    
    # 10 merchant categories
    merchant_categories = [
        'groceries', 'gas_station', 'dining', 'online_retail', 'travel',
        'entertainment', 'electronics', 'apparel', 'services', 'other'
    ]
    
    # Preferred categories (3 per account, seeded consistently based on account index)
    state = np.random.RandomState(acc_idx)
    pref = state.choice(merchant_categories, size=3, replace=False)
    
    # Assign category with 85% preferred, 15% non-preferred
    is_pref = (val // num_accounts) % 100 < 85
    non_pref = [c for c in merchant_categories if c not in pref]
    
    if is_pref:
        cat_idx = (val // (num_accounts * 100)) % len(pref)
        merchant_category = pref[cat_idx]
    else:
        cat_idx = (val // (num_accounts * 100)) % len(non_pref)
        merchant_category = non_pref[cat_idx]
        
    return account_id, merchant_category


if __name__ == "__main__":
    import os
    if not os.path.exists(csv_path):
        print(f"Error: Dataset '{csv_path}' not found in the current directory.")
    else:
        print(f"Loading dataset '{csv_path}'...")
        transactions = pd.read_csv(csv_path)
        print(f"Loaded {len(transactions)} rows.")

        # Augment with mock data columns for test block
        transactions = add_mock_identity_columns(transactions)

        # Select a sample flagged transaction (Class == 1)
        flagged_txs = transactions[transactions['Class'] == 1]
        if flagged_txs.empty:
            print("No flagged (Class == 1) transactions found in the dataset.")
        else:
            # Pick a sample transaction
            sample_index = flagged_txs.index[0]
            target_tx = transactions.loc[sample_index]
            account_id = target_tx['account_id']

            print("\n==================================================")
            print("TARGET TRANSACTION DETAILS (FLAGGED FOR FRAUD)")
            print("==================================================")
            print(f"Index:             {sample_index}")
            print(f"Account ID:        {account_id}")
            print(f"Time (seconds):    {target_tx['Time']}")
            print(f"Amount:            ${target_tx['Amount']:.2f}")
            print(f"Merchant Category: {target_tx['merchant_category']}")
            print(f"Actual Class:      {target_tx['Class']} (Fraud)")

            # Get historical profile deviation
            print("\nCalculating pattern deviation...")
            deviation = get_pattern_deviation(account_id, target_tx, transactions)
            print("Pattern Deviation Results:")
            for key, val in deviation.items():
                if val is None:
                    print(f"  {key}: None")
                elif isinstance(val, float):
                    print(f"  {key}: {val:.4f}")
                else:
                    print(f"  {key}: {val}")

            # Get recent account history
            print("\nRetrieving account history...")
            history = get_account_history(account_id, transactions, limit=5, before_time=target_tx['Time'])
            print(f"Account History (last 5 transactions prior to target transaction for {account_id}):")
            print(history[['Time', 'Amount', 'merchant_category', 'Class']])
