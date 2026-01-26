import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features.features import add_time_features, add_velocity_features

def test_add_time_features():
    df = pd.DataFrame({
        'timestamp': ['2023-01-01 10:00:00', '2023-01-02 11:30:00']
    })
    
    df_result = add_time_features(df)
    
    assert 'hour_of_day' in df_result.columns
    assert 'day_of_week' in df_result.columns
    assert df_result['hour_of_day'].iloc[0] == 10
    assert df_result['day_of_week'].iloc[0] == 6 # Sunday is 6

def test_add_velocity_features():
    # Create sample customer transactions
    df = pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2023-01-01 10:00:00',
            '2023-01-01 10:30:00', # Within 1h of first
            '2023-01-01 11:15:00'  # > 1h from first
        ]),
        'customer_id': ['C1', 'C1', 'C1'],
        'amount': [100.0, 50.0, 200.0]
    })
    
    # 1 Hour Window
    df_result = add_velocity_features(df, time_window_hours=1)
    
    # Check 2nd transaction (10:30): Should include 10:00 and 10:30
    # count_last_1h = 2
    # amount_last_1h = 150
    assert df_result['count_last_1h'].iloc[1] == 2
    assert df_result['amount_last_1h'].iloc[1] == 150.0
    
    # Check 3rd transaction (11:15): 
    # Window [10:15, 11:15]. 10:00 is out. 10:30 is in. 11:15 is in.
    # count_last_1h = 2
    # amount_last_1h = 50 + 200 = 250
    assert df_result['count_last_1h'].iloc[2] == 2
    assert df_result['amount_last_1h'].iloc[2] == 250.0

if __name__ == "__main__":
    pytest.main([__file__])
