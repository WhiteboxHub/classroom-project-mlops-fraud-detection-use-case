import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import os

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_data(n_transactions=10000):
    print(f"Generating {n_transactions} transactions...")
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=90)
    timestamps = [start_date + timedelta(minutes=random.randint(0, 90*24*60)) for _ in range(n_transactions)]
    timestamps.sort()
    
    # Generate Customer IDs
    n_customers = int(n_transactions / 10)
    customer_ids = [f"C{i:06d}" for i in np.random.randint(1, n_customers+1, n_transactions)]
    
    # Generate Merchant IDs
    n_merchants = int(n_transactions / 50)
    merchant_ids = [f"M{i:06d}" for i in np.random.randint(1, n_merchants+1, n_transactions)]
    
    # Transaction Amounts (Log-normal distribution)
    amounts = np.random.lognormal(mean=4.0, sigma=1.0, size=n_transactions)
    amounts = np.round(amounts, 2)
    
    # Locations (simplified as simple coordinates 0-100)
    # We will assume home location is roughly consistent for legit, weird for fraud
    # Just generating random coordinates for now
    lat = np.random.uniform(0, 100, n_transactions)
    long = np.random.uniform(0, 100, n_transactions)
    
    data = pd.DataFrame({
        "timestamp": timestamps,
        "customer_id": customer_ids,
        "merchant_id": merchant_ids,
        "amount": amounts,
        "lat": lat,
        "long": long    
    })
    
    # Simulate Fraud
    # Fraud Patterns:
    # 1. High amount
    # 2. Rapid succession (velocity) - harder to simulate simply without loop, 
    #    so we'll just probabilistic assignment based on amount for 'simple' fraud
    #    and some random 'complex' fraud.
    
    data['is_fraud'] = 0
    
    # P(Fraud | High Amount) > P(Fraud | Low Amount)
    high_amount_mask = data['amount'] > 500
    data.loc[high_amount_mask, 'is_fraud'] = np.random.choice([0, 1], size=high_amount_mask.sum(), p=[0.8, 0.2])
    
    # Random low amount fraud
    data.loc[~high_amount_mask, 'is_fraud'] = np.random.choice([0, 1], size=(~high_amount_mask).sum(), p=[0.99, 0.01])
    
    # Save to CSV
    output_path = "data/raw/transactions.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(data['is_fraud'].value_counts())

if __name__ == "__main__":
    generate_data()
