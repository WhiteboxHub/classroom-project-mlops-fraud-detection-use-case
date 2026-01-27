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
- **Python 3.9+** (Ensure it's added to your PATH)
- **Docker Desktop** (Make sure it is running)
- **Git**

### 2. Installation

Open your terminal (PowerShell or Command Prompt) and run:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Data
Create synthetic transaction data mimicking fraud patterns. This script creates `data/raw/transactions.csv`.

```bash
python src/utils/generate_data.py

# Initialize DVC (first time only)
dvc init
dvc add data/raw/transactions.csv
```

### 4. Run Tests
Verify that the feature engineering logic functions correctly:

```bash
pytest tests/
```

### 5. Train Model
Run the training pipeline. This script loads data, calculates features, performs 5-Fold Cross Validation, and logs metrics to MLflow.

```bash
python src/models/train.py
```
*You can examine the results in the MLflow UI (after starting the full stack below).*

### 6. Drift Detection
Run a simulation to check for data drift between training data and new incoming data:

```bash
python src/monitoring/drift.py
```

### 7. Run Inference Service (Full Stack)
Start the entire MLOps stack, including the API, MLflow server, and Redis feature store.

```bash
docker-compose up --build
```

**Access Services:**
- **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow UI**: [http://localhost:5000](http://localhost:5000)

---

### 📩 Test Prediction

You have multiple ways to test the API once the system is running:

#### 1. Swagger UI (Easiest)
Interact with the API directly from your browser:
1.  Navigate to [http://localhost:8000/docs](http://localhost:8000/docs).
2.  Expand the `POST /predict` endpoint.
3.  Click **Try it out**.
4.  Paste the following JSON into the Request Body:
```json
{
  "timestamp": "2023-10-27T10:00:00",
  "customer_id": "C999999",
  "merchant_id": "M_SCAM",
  "amount": 9000.0,
  "lat": 45.0,
  "long": 45.0
}
```
5.  Click **Execute** and check the response.

#### 2. Windows PowerShell
Open a new terminal window:
```powershell
$body = @{
    timestamp = "2023-10-27T10:00:00",
    customer_id = "C999999",
    merchant_id = "M_SCAM",
    amount = 9000.0,
    lat = 45.0,
    long = 45.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

#### 3. Sample Data
Use these JSON snippets for testing in Swagger UI or Postman:

**High Value Fraud Pattern:**
```json
{
  "timestamp": "2023-10-27T12:00:00",
  "customer_id": "C_TEST_01",
  "merchant_id": "M_TEST_01",
  "amount": 5500.00,
  "lat": 40.71,
  "long": -74.00
}
```

**Normal Transaction (Low Amount):**
```json
{
  "timestamp": "2023-10-27T12:05:00",
  "customer_id": "C_TEST_02",
  "merchant_id": "M_TEST_02",
  "amount": 25.50,
  "lat": 40.71,
  "long": -74.00
}
```

#### 4. Troubleshooting
- **Connection Refused**: Ensure `docker-compose` is running.
- **Internal Server Error**: Check the Docker terminal output for Python errors.
- **Verify Training**: Check [http://localhost:5000](http://localhost:5000) for MLflow experiments.
- **Dependency Mismatch**: If you see `InconsistentVersionWarning`, ensure local and Docker use the same `scikit-learn` version (currently `1.6.1`). Fix by running `pip install -r requirements.txt` locally and retraining.

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
   - `train.py` reads local DVC-tracked data.
   - Feature Store saves offline parquet files for consistent training data.
   - Model and metrics are logged to MLflow experiment `fraud_detection_baseline`.

3. **Deployment**:
   - `app.py` loads the latest Production model from MLflow.
   - Retrieves online features from Redis (populated by ingestion pipeline - mocked for this Phase 1 POC).
