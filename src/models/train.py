import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.features.features import calculate_features

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment("fraud_detection_xgb")

ENSEMBLE_THRESHOLD = 0.6  #  SAME threshold used in serving

print("Loading data...")
df = pd.read_csv("data/raw/transactions.csv")

print("Calculating features...")
df_feat = calculate_features(df)

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

X = df_feat[FEATURES]
y = df_feat["is_fraud"]

scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

# ---------------- MODELS ----------------
lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=500
)

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    max_depth=5,
    n_estimators=200,
    learning_rate=0.08,
    random_state=42
)

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", lr_model)
])

xgb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", xgb_model)
])

with mlflow.start_run():
    lr_pipeline.fit(X, y)
    xgb_pipeline.fit(X, y)

    lr_prob = lr_pipeline.predict_proba(X)[:, 1]
    xgb_prob = xgb_pipeline.predict_proba(X)[:, 1]

    ensemble_prob = 0.5 * lr_prob + 0.5 * xgb_prob
    ensemble_pred = (ensemble_prob >= ENSEMBLE_THRESHOLD).astype(int)

    mlflow.log_metric("precision", precision_score(y, ensemble_pred))
    mlflow.log_metric("recall", recall_score(y, ensemble_pred))
    mlflow.log_metric("f1", f1_score(y, ensemble_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y, ensemble_prob))

    mlflow.log_param("ensemble_threshold", ENSEMBLE_THRESHOLD)

    mlflow.sklearn.log_model(lr_pipeline, "lr_model")
    mlflow.sklearn.log_model(xgb_pipeline, "xgb_model")

print("Both models trained and logged to MLflow")
