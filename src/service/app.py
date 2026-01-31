from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
import os
import sys
import mlflow
import mlflow.pyfunc
import glob
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.features.feature_store import FeatureStore
from src.features.features import add_time_features
from src.service.schemas import TransactionRequest, PredictionResponse

# Global variables
model = None
feature_store = None
RUN_ID = None


def load_latest_model():
    """
    Loads the latest MLflow model from the models/ directory
    (because this project stores models there, not in run artifacts).
    """
    MLFLOW_PATH = os.getenv("MLFLOW_TRACKING_DIR", "/mlruns")
    mlflow.set_tracking_uri(f"file:{MLFLOW_PATH}")

    print("Using MLflow path:", MLFLOW_PATH)

    # Find experiment folder
    exp_dirs = glob.glob(os.path.join(MLFLOW_PATH, "*"))
    if not exp_dirs:
        raise Exception("No experiments found in mlruns.")

    # Take latest experiment
    exp_dir = sorted(exp_dirs, key=os.path.getmtime)[-1]

    models_dir = os.path.join(exp_dir, "models")
    if not os.path.exists(models_dir):
        raise Exception("No models directory found in mlruns experiment.")

    model_versions = glob.glob(os.path.join(models_dir, "m-*"))
    if not model_versions:
        raise Exception("No model versions found in models directory.")

    latest_model_dir = sorted(model_versions, key=os.path.getmtime)[-1]
    model_artifact_path = os.path.join(latest_model_dir, "artifacts")

    print(f"Loading model from {model_artifact_path}...")

    loaded_model = mlflow.pyfunc.load_model(model_artifact_path)

    return loaded_model, os.path.basename(latest_model_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_store, RUN_ID
    try:
        model, RUN_ID = load_latest_model()
        feature_store = FeatureStore(redis_host=os.getenv("REDIS_HOST", "localhost"))
        print("Model and Feature Store initialized.")
    except Exception as e:
        print("FATAL: Model could not be loaded")
        print(e)
        raise e
    yield


app = FastAPI(title="Fraud Detection API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_version": RUN_ID}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Fetch Online Features from Feature Store
    customer_features = feature_store.get_online_features(request.customer_id)

    if not customer_features:
        customer_features = {
            "count_last_1h": 0,
            "amount_last_1h": 0,
            "count_last_24h": 0,
            "amount_last_24h": 0
        }

    # 2. Compute Real-time Features
    tx_data = request.dict()
    df = pd.DataFrame([tx_data])

    df = add_time_features(df)

    for k, v in customer_features.items():
        if k not in df.columns:
            df[k] = v

    required_features = [
        'amount', 'lat', 'long', 'hour_of_day', 'day_of_week',
        'count_last_1h', 'amount_last_1h', 'count_last_24h', 'amount_last_24h'
    ]

    for feat in required_features:
        if feat not in df.columns:
            df[feat] = 0.0

    X = df[required_features]

    # ✅ PyFunc prediction
    y_pred = model.predict(X)

    if isinstance(y_pred, (list, tuple, np.ndarray, pd.Series)):
        prob = float(y_pred[0])
    else:
        prob = float(y_pred)

    prediction = "FRAUD" if prob > 0.5 else "LEGIT"

    #   Safe explanation for PyFunc model
    explanation = {
    "model_confidence": float(prob)
    } 


    return PredictionResponse(
        prediction=prediction,
        probability=prob,
        explanation=explanation,
        model_version=RUN_ID
    )
