import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import os

np.random.seed(42)
random.seed(42)

NUM_CUSTOMERS = 500
NUM_TRANSACTIONS = 10000

customers = [f"C{i:06d}" for i in range(NUM_CUSTOMERS)]
merchants = ["Amazon", "Flipkart", "Hospital", "Crypto", "Electronics", "Grocery"]

merchant_risk = {
    "Amazon": 0.05,
    "Flipkart": 0.05,
    "Grocery": 0.03,
    "Hospital": 0.02,
    "Electronics": 0.2,
    "Crypto": 0.6
}

customer_profiles = {
    c: {
        "avg_amount": np.random.uniform(200, 2000),
        "home_lat": np.random.uniform(10, 20),
        "home_long": np.random.uniform(70, 80)
    } for c in customers
}

rows = []
current_time = datetime.now()

for _ in range(NUM_TRANSACTIONS):
    cust = random.choice(customers)
    profile = customer_profiles[cust]

    merchant = random.choice(merchants)
    risk = merchant_risk[merchant]

    is_night = random.random() < 0.2
    hour = random.randint(0, 5) if is_night else random.randint(8, 21)

    amount = abs(np.random.normal(profile["avg_amount"], profile["avg_amount"] * 0.6))

    far_location = random.random() < 0.05
    lat = profile["home_lat"] + (random.uniform(20, 40) if far_location else random.uniform(-0.1, 0.1))
    long = profile["home_long"] + (random.uniform(20, 40) if far_location else random.uniform(-0.1, 0.1))

    fraud_score = 0
    if amount > profile["avg_amount"] * 3:
        fraud_score += 1
    if risk > 0.5:
        fraud_score += 1
    if far_location:
        fraud_score += 1
    if is_night:
        fraud_score += 1

    is_fraud = 1 if fraud_score >= 2 else 0

    rows.append([
        current_time.strftime("%Y-%m-%d %H:%M:%S"),
        cust,
        merchant,
        round(amount, 2),
        lat,
        long,
        is_fraud
    ])

    current_time += timedelta(seconds=random.randint(30, 300))

df = pd.DataFrame(rows, columns=[
    "timestamp", "customer_id", "merchant_id", "amount", "lat", "long", "is_fraud"
])

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/transactions.csv", index=False)
print(df["is_fraud"].value_counts())
