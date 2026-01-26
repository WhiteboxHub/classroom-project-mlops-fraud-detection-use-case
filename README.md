# MLOps Fraud Detection System - Phase 1

This repository contains an end-to-end MLOps implementation for critical fraud detection, adhering to strict operational requirements for latency, reproducibility, and explainability.

## 🚀 Features

- **Offline Training Pipeline**:
  - Reproducible data with **DVC**.
  - Experiment tracking with **MLflow**.
  - **Stratified K-Fold Cross Validation** for robust evaluation.
  - Logistic Regression with class weighting for imbalanced data.

- **Feature Store**:
  - Consistent feature logic for training (Batch/Parquet) and inference (Online/Redis).
  - Time-Window features (Velocity: `count_last_1h`, `amount_last_24h`).

- **Inference Service**:
  - **FastAPI** application serving predictions < 100ms.
  - Local Explainability (Top contributing features).
  - Pydantic validation.

- **Monitoring & Observability**:
  - **Drift Detection** using Kolmogorov-Smirnov (KS) test.
  - Dockerized stack for easy deployment.

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **Frameworks**: FastAPI, scikit-learn, pandas
- **MLOps**: MLflow, DVC, Docker, Redis

## 📂 Project Structure

```bash
.
├── data/                   # Data storage (DVC tracked)
├── infrastructure/         # Docker configs
├── src/
│   ├── features/           # Feature engineering logic
│   ├── models/             # Training scripts
│   ├── monitoring/         # Drift detection
│   ├── service/            # FastAPI app
│   └── utils/              # Data generation
├── tests/                  # Unit tests
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## ⚡ Quick Start (Local)

### 1. Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Git

### 2. Installation
```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Data
Create synthetic transaction data mimicking fraud patterns:
```bash
python src/utils/generate_data.py
# Initialize DVC (if not already)
dvc init
dvc add data/raw/transactions.csv
```

### 4. Run Tests
Verify feature engineering logic:
```bash
pytest tests/
```

### 5. Train Model
Run the training pipeline with K-Fold validation:
```bash
python src/models/train.py
```
*Check `./mlruns` (or `mlflow ui`) to see experiments.*

### 6. Drift Detection
Simulate and check for data drift:
```bash
python src/monitoring/drift.py
```

### 7. Run Inference Service
Start the full stack (API + MLflow + Redis):
```bash
docker-compose up --build
```

**Test Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "timestamp": "2023-10-27T10:00:00",
  "customer_id": "C123456", 
  "merchant_id": "M_TEST", 
  "amount": 5000.0, 
  "lat": 50.0, 
  "long": 50.0
}'
```

**Expected Response**:
```json
{
  "prediction": "FRAUD",
  "probability": 1.0,
  "explanation": { ... },
  "model_version": "..."
}
```

## 📋 MLOps Workflow

1. **Development**:
   - Data scientists work in `notebooks/` and refactor logic to `src/features/`.
   - Run `pytest` to ensure logic validity.

2. **Training**:
   - `train.py` pulls data via DVC (automating `dvc pull`).
   - Feature Store saves offline parquet.
   - Model and metrics logged to MLflow under `fraud_detection_baseline`.

3. **Deployment**:
   - `app.py` loads the latest Production model from MLflow.
   - Retrieves online features from Redis (populated by ingestion pipeline - mocked for POC).
