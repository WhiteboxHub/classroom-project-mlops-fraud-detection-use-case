import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import sys
import os

def detect_drift(reference_path="data/raw/transactions.csv", current_path=None, threshold=0.05):
    """
    Compares two datasets and detects feature drift using KS Test.
    reference_path: Path to training data (baseline).
    current_path: Path to new inference data. If None, we simulate drift for demo.
    """
    print("Loading reference data...")
    ref_df = pd.read_csv(reference_path)
    
    if current_path:
        curr_df = pd.read_csv(current_path)
    else:
        print("Simulating current data (with drift)...")
        # Simulate drift: Amount increases significantly
        curr_df = ref_df.copy()
        curr_df['amount'] = curr_df['amount'] * 1.5 + np.random.normal(0, 10, len(curr_df))
    
    numerical_features = ['amount', 'lat', 'long']
    
    drift_detected = False
    print(f"\nChecking for Drift (KS Test, p-value < {threshold})...")
    
    for feature in numerical_features:
        # KS Test
        stat, p_value = ks_2samp(ref_df[feature], curr_df[feature])
        
        is_drift = p_value < threshold
        status = "DRIFT DETECTED!" if is_drift else "No Drift"
        
        print(f"Feature '{feature}': p-value={p_value:.5f} -> {status}")
        
        if is_drift:
            drift_detected = True
            
    if drift_detected:
        print("\nALERT: Data Drift detected requiring attention!")
    else:
        print("\nSystem status: Healthy")

if __name__ == "__main__":
    detect_drift()
