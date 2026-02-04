import pandas as pd

def add_time_features(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_night"] = (df["hour_of_day"] < 6).astype(int)
    return df

def calculate_features(df):
    df = df.sort_values(["customer_id", "timestamp"])
    df = add_time_features(df)

    df["count_last_1h"] = df.groupby("customer_id")["timestamp"].transform(
        lambda x: x.rolling(5, min_periods=1).count()
    )

    df["amount_last_1h"] = df.groupby("customer_id")["amount"].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )

    df["avg_amount_7d"] = df.groupby("customer_id")["amount"].transform(
        lambda x: x.rolling(50, min_periods=1).mean()
    )

    df["amount_ratio"] = df["amount"] / df["avg_amount_7d"]

    return df
