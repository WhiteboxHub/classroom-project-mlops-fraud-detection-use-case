from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
import mlflow
import mlflow.sklearn
import os, sys, time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.features.feature_store import FeatureStore
from src.features.features import add_time_features
from src.service.schemas import TransactionRequest, PredictionResponse

lr_model = None
xgb_model = None
feature_store = None
RUN_ID = None

ENSEMBLE_THRESHOLD = 0.6   # SAME as training

FEATURES = [
    "amount",
    "lat",
    "long",
    "hour_of_day",
    "day_of_week",
    "is_night",
    "count_last_1h",
    "amount_last_1h",
    "avg_amount_7d",
    "amount_ratio"
]

# ---------------- LOAD MODELS ----------------
def load_latest_model():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

    EXP_NAME = "fraud_detection_xgb"
    exp = mlflow.get_experiment_by_name(EXP_NAME)
    if exp is None:
        raise Exception("Experiment not created yet")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time desc"],
        max_results=1
    )

    if runs.empty:
        raise Exception("No runs in experiment yet")

    run_id = runs.iloc[0]["run_id"]

    lr_uri = f"runs:/{run_id}/lr_model"
    xgb_uri = f"runs:/{run_id}/xgb_model"

    print("Loading LR model from:", lr_uri)
    print("Loading XGB model from:", xgb_uri)

    lr_model = mlflow.sklearn.load_model(lr_uri)
    xgb_model = mlflow.sklearn.load_model(xgb_uri)

    return lr_model, xgb_model, run_id


# ---------------- LIFESPAN ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global lr_model, xgb_model, feature_store, RUN_ID

    for i in range(60):
        try:
            lr_model, xgb_model, RUN_ID = load_latest_model()
            print("Models loaded successfully")
            break
        except Exception as e:
            print(f"Waiting for models... ({i})", e)
            time.sleep(2)

    if lr_model is None or xgb_model is None:
        raise RuntimeError("Models could not be loaded")

    feature_store = FeatureStore(redis_host=os.getenv("REDIS_HOST", "redis"))
    print("Models and Feature Store initialized.")
    yield


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)


# ---------------- HEALTH ----------------
@app.get("/health")
def health_check():
    return {"status": "ok", "model_version": RUN_ID}


# ---------------- PREDICT ----------------
@app.post("/predict", response_model=PredictionResponse)
def predict(req: TransactionRequest):
    if lr_model is None or xgb_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    customer_features = feature_store.get_online_features(req.customer_id)

    has_history = customer_features is not None  # TRUE cold-start check

    df = pd.DataFrame([req.dict()])
    df = add_time_features(df)

    if customer_features:
        for k, v in customer_features.items():
            if k not in df.columns:
                df[k] = v

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    df["avg_amount_7d"] = df["avg_amount_7d"].fillna(df["amount"])
    df.loc[df["avg_amount_7d"] == 0, "avg_amount_7d"] = df["amount"]

    df["amount_ratio"] = df["amount"] / df["avg_amount_7d"]

    X = df[FEATURES]

    lr_prob = float(lr_model.predict_proba(X)[0][1])
    xgb_prob = float(xgb_model.predict_proba(X)[0][1])

    final_prob = 0.5 * lr_prob + 0.5 * xgb_prob

    # ---------- RULES FIRST ----------
    if df["amount_ratio"].iloc[0] > 3:
        prediction = "FRAUD"
        reason = "amount_spike"

    elif df["is_night"].iloc[0] == 1 and df["amount"].iloc[0] > 2000:
        prediction = "FRAUD"
        reason = "night_high_value"

    # ---------- ML ONLY IF HISTORY ----------
    elif has_history and final_prob >= ENSEMBLE_THRESHOLD:
        prediction = "FRAUD"
        reason = "ml_probability"

    else:
        prediction = "LEGIT"
        reason = "normal_behavior"

    df["timestamp"] = df["timestamp"].astype(str)
    feature_store.save_online(df)

    return PredictionResponse(
        prediction=prediction,
        probability=final_prob,
        explanation={
            "model_confidence": final_prob,
            "decision_reason": reason
        },
        model_version=RUN_ID
    )
