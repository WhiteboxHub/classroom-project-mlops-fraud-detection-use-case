import redis
import json

class FeatureStore:
    def __init__(self, redis_host="redis", redis_port=6379):
        self.client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    def save_online(self, df):
        latest = df.groupby("customer_id").last().reset_index()

        for _, row in latest.iterrows():
            data = row.to_dict()

            # Convert Timestamp to string
            if "timestamp" in data:
                data["timestamp"] = str(data["timestamp"])

            self.client.set(
                f"customer:{row.customer_id}",
                json.dumps(data)
            )

    def get_online_features(self, customer_id):
        data = self.client.get(f"customer:{customer_id}")
        return json.loads(data) if data else None
