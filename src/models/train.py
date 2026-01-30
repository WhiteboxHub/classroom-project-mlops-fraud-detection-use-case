import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.features.features import calculate_features
from src.features.feature_store import FeatureStore

def train():
    MLFLOW_DIR = os.getenv("MLFLOW_TRACKING_DIR", "./mlruns")
    mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
    mlflow.set_experiment("fraud_detection_baseline")

    with mlflow.start_run() as run:
        print("Loading data...")
        df = pd.read_csv("data/raw/transactions.csv")
        
        print("Calculating features...")
        df_features = calculate_features(df)
        
        # Save features to offline store
        fs = FeatureStore()
        fs.save_offline(df_features, "training_features")
        
        target = 'is_fraud'
        drop_cols = ['timestamp', 'customer_id', 'merchant_id', 'is_fraud']
        
        # Prepare X and y
        X = df_features.drop(columns=drop_cols)
        y = df_features[target]
        
        print(f"Total samples: {len(X)}")
        print(f"Features: {X.columns.tolist()}")
        
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("cv_folds", 5)
        
        # Define Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0))
        ])
        
        # Stratified K-Fold
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        metrics = {"precision": [], "recall": [], "f1": [], "roc_auc": [], "pr_auc": []}
        
        print("Starting 5-Fold Cross Validation...")
        fold = 1
        for train_index, test_index in skf.split(X, y):
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            
            pipeline.fit(X_train_fold, y_train_fold)
            
            y_pred = pipeline.predict(X_test_fold)
            y_prob = pipeline.predict_proba(X_test_fold)[:, 1]
            
            metrics["precision"].append(precision_score(y_test_fold, y_pred))
            metrics["recall"].append(recall_score(y_test_fold, y_pred))
            metrics["f1"].append(f1_score(y_test_fold, y_pred))
            metrics["roc_auc"].append(roc_auc_score(y_test_fold, y_prob))
            metrics["pr_auc"].append(average_precision_score(y_test_fold, y_prob))
            
            print(f"Fold {fold} - ROC AUC: {metrics['roc_auc'][-1]:.4f}")
            fold += 1
            
        # Log Average Metrics
        for metric, values in metrics.items():
            avg_val = np.mean(values)
            mlflow.log_metric(f"avg_{metric}", avg_val)
            print(f"Avg {metric}: {avg_val:.4f}")
            
        # Retrain on Full Dataset for Production
        print("Retraining on full dataset...")
        pipeline.fit(X, y)
        
        # Log Model
        mlflow.sklearn.log_model(pipeline, "model")
        print("Final model saved to MLflow.")

if __name__ == "__main__":
    train()
