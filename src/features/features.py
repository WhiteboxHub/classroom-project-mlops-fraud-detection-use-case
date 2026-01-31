import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time-based features:
    - hour_of_day
    - day_of_week
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    return df

def add_velocity_features(df: pd.DataFrame, time_window_hours: int = 1) -> pd.DataFrame:
    """
    Adds velocity features (counts/sums over a window) for each customer.
    For offline training (batch).
    """
    # specific to offline batch processing with full history
    df = df.sort_values(['customer_id', 'timestamp'])
    
    # We use grouping and rolling windows
    # Since rolling is based on index, we set timestamp as index
    df_indexed = df.set_index('timestamp')
    
    # Group by customer
    grouped = df_indexed.groupby('customer_id')
    
    # Count of transactions in last X hours
    # '1h' requires pandas timedelta index
    feature_name = f'count_last_{time_window_hours}h'
    
    # Note: excluding the current row implies 'closed="left"', but pandas default includes current.
    # In fraud, usually we include current attempt or exclude? 
    # Usually we want "history BEFORE this tx". 
    # But for simplicity in pandas rolling, we often include current.
    # Let's subtract 1 if we include current, or assume feature is "velocity INCLUDING current".
    # We'll stick to standard rolling which includes current point.
    
    df[feature_name] = grouped['amount'].rolling(f'{time_window_hours}h').count().values
    
    # Sum of amount in last X hours
    feature_name_sum = f'amount_last_{time_window_hours}h'
    df[feature_name_sum] = grouped['amount'].rolling(f'{time_window_hours}h').sum().values
    
    # Mean amount in last X hours
    # df[f'mean_last_{time_window_hours}h'] = grouped['amount'].rolling(f'{time_window_hours}h').mean().values
    
    return df

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main pipeline to calculate all features.
    """
    df = add_time_features(df)
    df = add_velocity_features(df, time_window_hours=1)
    df = add_velocity_features(df, time_window_hours=24)
    return df
