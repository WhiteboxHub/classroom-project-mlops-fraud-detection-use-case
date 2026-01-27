from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import mlflow.sklearn
import pandas as pd
import os
import sys

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
    Finds the latest run in the 'fraud_detection_baseline' experiment and loads the model.
    """
    mlflow.set_tracking_uri("file:./mlruns")
    experiment = mlflow.get_experiment_by_name("fraud_detection_baseline")
    if not experiment:
        raise Exception("Experiment 'fraud_detection_baseline' not found. Run training first.")
    
    # Search runs, order by start_time desc
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if runs.empty:
        raise Exception("No runs found in experiment.")
    
    run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from {model_uri}...")
    
    try:
        loaded_model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Fallback 1: Try local mlruns path
        try:
            print("Standard load failed. Attempting fallback via local path...")
            artifact_path = os.path.join("mlruns", experiment.experiment_id, run_id, "artifacts", "model")
            loaded_model = mlflow.sklearn.load_model(artifact_path)
        except Exception:
            # Fallback 2: Try explicit local storage
            print("MLruns fallback failed. Attempting model_storage...")
            loaded_model = mlflow.sklearn.load_model("model_storage")

    return loaded_model, run_id

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, feature_store, RUN_ID
    try:
        model, RUN_ID = load_latest_model()
        feature_store = FeatureStore(redis_host=os.getenv("REDIS_HOST", "localhost"))
        print("Model and Feature Store initialized.")
    except Exception as e:
        print(f"Error initializing app: {e}")
        # We might want to fail hard here in prod, but for dev we warn
        pass
    yield
    # Shutdown

app = FastAPI(title="Fraud Detection API", version="0.1.0", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_version": RUN_ID}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 1. Fetch Online Features from Feature Store
    # In a real system, we'd fetch historical aggregates for this customer (e.g. velocity)
    # For this POC, we'll try to fetch from Redis, or if missing/mocked, we might use zeros or simple logic.
    # To demonstrate functionality, we'll implement a fallback if Redis is empty or down.
    
    customer_features = feature_store.get_online_features(request.customer_id)
    
    # If not found (new customer or store empty), initialize default
    if not customer_features:
        # Fallback: treat as fresh customer
        # For velocity features (count_last_1h, amount_last_1h...), set to 0
        customer_features = {
            "count_last_1h": 0,
            "amount_last_1h": 0,
            "count_last_24h": 0, # Assuming we want 24h too
            "amount_last_24h": 0
        }
    
    # 2. Compute Real-time Features (Time of day, etc)
    # We construct a DataFrame because our transformers expect it
    tx_data = request.dict()
    df = pd.DataFrame([tx_data])
    
    # Add time features
    df = add_time_features(df)
    
    # Merge with online features
    # Note: simple assignment works because single row
    for k, v in customer_features.items():
        if k not in df.columns:
            df[k] = v
            
    # Calculate additional on-the-fly 'velocity' update if needed?
    # For simplicity, we just use the retrieved history features + current tx numeric features.
    
    # 3. Predict
    # Ensure columns match training
    # We need to know the columns.
    # Usually the model wrapper handles this or we stored the signature.
    # For this POC, we assume the pipeline handles scaling.
    # We need to ensure we have the columns:
    # ['amount', 'lat', 'long', 'hour_of_day', 'day_of_week', 'count_last_1h', 'amount_last_1h', 'count_last_24h', 'amount_last_24h']
    # Wait, 'count_last_24h' was calculated in features.py: `add_velocity_features(df, time_window_hours=24)`
    # Which creates 'count_last_24h' and 'amount_last_24h'.
    
    # Ensure they exist (if Redis missed them)
    required_features = ['amount', 'lat', 'long', 'hour_of_day', 'day_of_week', 
                         'count_last_1h', 'amount_last_1h', 'count_last_24h', 'amount_last_24h']
    
    for feat in required_features:
        if feat not in df.columns:
            df[feat] = 0.0 # Impute missing
            
    # Select only required columns in order (Optional, but safe)
    X = df[required_features]
    
    # Predict
    prob = model.predict_proba(X)[0][1]
    prediction = "FRAUD" if prob > 0.5 else "LEGIT"
    
    # Explanation (Top features)
    # Logistic Regression has 'model' step in pipeline.
    # pipeline.steps[-1][1] is the model.
    # coefficients = ...
    # We can compute contribution: val * coef
    
    explanation = {}
    try:
        classifier = model.named_steps['model']
        scaler = model.named_steps['scaler']
        
        # Get feature names if possible, else use list
        feature_names = required_features
        
        # Coefficients
        coefs = classifier.coef_[0]
        
        # Transformed input (scaled)
        X_scaled = scaler.transform(X)[0]
        
        contributions = {}
        for name, val, coef in zip(feature_names, X_scaled, coefs):
            contributions[name] = abs(val * coef) # Magnitude of contribution
            
        # Top 3
        sorted_contribs = sorted(contributions.items(), key=lambda item: item[1], reverse=True)[:3]
        explanation = dict(sorted_contribs)
        
    except Exception as e:
        print(f"Explanation error: {e}")
        explanation = {"error": "Could not explain"}
    
    return PredictionResponse(
        prediction=prediction,
        probability=float(prob),
        explanation=explanation,
        model_version=RUN_ID
    )
