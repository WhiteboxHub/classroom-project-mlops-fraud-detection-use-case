import pandas as pd
import redis
import json
import os
from typing import Dict, Any, Optional

class FeatureStore:
    def __init__(self, offline_path: str = "data/features", redis_host: str = "localhost", redis_port: int = 6379):
        self.offline_path = offline_path
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Determine if we are in "docker" (usually we might set env var)
        # For now, we try to connect, if fail, we might warn or use mock
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True, socket_connect_timeout=1)
            self.redis_client.ping()
        except (redis.ConnectionError, redis.exceptions.TimeoutError):
            print("Warning: Could not connect to Redis. Online features will not work/be saved.")
            self.redis_client = None

    def save_offline(self, df: pd.DataFrame, name: str):
        """
        Saves features to parquet for training.
        """
        os.makedirs(self.offline_path, exist_ok=True)
        path = os.path.join(self.offline_path, f"{name}.parquet")
        df.to_parquet(path)
        print(f"Saved offline features to {path}")

    def save_online(self, df: pd.DataFrame):
        """
        Saves the LATEST feature values for each customer to Redis.
        Key = customer_id
        Value = JSON of features
        """
        if not self.redis_client:
            return

        print("Saving online features to Redis...")
        # For each customer, get the last row
        # Assuming df is sorted by time
        latest_df = df.groupby('customer_id').last().reset_index()
        
        pipeline = self.redis_client.pipeline()
        for _, row in latest_df.iterrows():
            key = f"customer_features:{row['customer_id']}"
            data = row.to_dict()
            # Convert timestamp to string
            if 'timestamp' in data:
                data['timestamp'] = str(data['timestamp'])
            
            pipeline.set(key, json.dumps(data))
        
        pipeline.execute()
        print(f"Saved {len(latest_df)} customer profiles to Redis.")

    def get_online_features(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve latest features from Redis.
        """
        if not self.redis_client:
            return None
        
        data = self.redis_client.get(f"customer_features:{customer_id}")
        if data:
            return json.loads(data)
        return None
