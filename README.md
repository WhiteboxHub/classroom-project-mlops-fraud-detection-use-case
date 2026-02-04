# 🛡️ MLOps Fraud Detection System – Phase 1 (Real-Time, Ensemble Model)

This repository contains an end-to-end **MLOps fraud detection system** designed to simulate **real-world banking fraud detection** with:

- Offline training  
- Online feature store (Redis)  
- Real-time inference (FastAPI)  
- MLflow model tracking  
- Dockerized deployment  

The system follows an **industry-style architecture**:

Batch training → Feature store → Model registry → Real-time inference

---

## 🚀 Features

### ✅ Offline Training Pipeline
- Synthetic transaction data generation with realistic fraud patterns.
- Feature engineering:
  - **Velocity features**: `count_last_1h`, `amount_last_1h`
  - **Behavioral features**: `avg_amount_7d`, `amount_ratio`
  - **Time features**: `hour_of_day`, `day_of_week`, `is_night`
- Class-imbalanced handling using `scale_pos_weight`.
- **Two models are trained together**:
  - Logistic Regression (baseline, stable linear model)
  - XGBoost (non-linear boosted tree model)
- Metrics logged to **MLflow**:
  - Precision  
  - Recall  
  - F1-score  
  - ROC-AUC  

Both models are logged in the same MLflow run as:
- `lr_model`
- `xgb_model`

---

### 🧠 Feature Store (Offline + Online)
- **Offline (training time)**:
  - Features are computed using `calculate_features()`:
    - Time features from timestamp
    - Rolling transaction counts and sums
    - Customer behavioral averages
- **Online (inference time)**:
  - Redis stores the **latest customer profile**:
    - `avg_amount_7d`
    - `count_last_1h`
    - `amount_last_1h`
    - `amount_ratio`
- Same feature logic is used for:
  - Training
  - Inference  
This guarantees **feature consistency** (no training-serving skew).

---

### ⚡ Inference Service (FastAPI)
- REST API for fraud prediction
- Loads latest MLflow models automatically:
  - Logistic Regression model
  - XGBoost model
- Fetches customer history from Redis
- Computes real-time features
- Produces:
  - Prediction (`FRAUD` or `LEGIT`)
  - Fraud probability
  - Explanation
  - Model version

---

### 🤖 Ensemble Prediction Logic

The system uses a **hybrid decision strategy**:

#### 1️⃣ ML Ensemble Probability

```
lr_prob = LogisticRegression.predict_proba()
xgb_prob = XGBoost.predict_proba()

final_prob = 0.5 * lr_prob + 0.5 * xgb_prob
```

#### 2️⃣ Rule-Based Safety Layer (Business Logic)

Certain fraud rules always override ML:
- If `amount_ratio > 3` → FRAUD  
- If `is_night == 1` and `amount > 2000` → FRAUD  

#### 3️⃣ Final Decision
- If rule-based fraud → `FRAUD`
- Else if ensemble probability ≥ threshold → `FRAUD`
- Else → `LEGIT`

This mimics **real bank systems** where:
ML + deterministic rules are combined.

---

### 🐳 Dockerized Stack
- FastAPI (Inference API)
- MLflow (Model tracking)
- Redis (Online feature store)
- Trainer service (auto-trains on startup)

---

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **ML**: scikit-learn, XGBoost, pandas
- **API**: FastAPI, Pydantic
- **MLOps**: MLflow, Redis, Docker, Docker Compose

---

## 📂 Project Structure

```bash
.
├── data/
│   └── raw/
│       └── transactions.csv
├── infrastructure/
│   └── Dockerfile.api
├── src/
│   ├── features/
│   │   ├── features.py
│   │   └── feature_store.py
│   ├── models/
│   │   └── train.py
│   ├── service/
│   │   ├── app.py
│   │   └── schemas.py
│   └── utils/
│       └── generate_data.py
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

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
```

### 3️⃣ Generate Data
```bash
python src/utils/generate_data.py
```

---

### 4️⃣ Train Models (Local)
```bash
python src/models/train.py
```
This trains:
- Logistic Regression model  
- XGBoost model  
and logs both to MLflow.

---

### 5️⃣ Run Full Stack (Docker)
```bash
docker compose up --build
```

This starts:
- Redis
- MLflow
- Trainer service
- Fraud API

---

## 🔍 API Endpoints

### Health
```http
GET /health
```

### Predict
```http
POST /predict
```

---

## 🧪 Testing Payloads

### ✅ LEGIT
```json
{
  "timestamp": "2026-02-06T10:00:00",
  "customer_id": "C777777",
  "merchant_id": "Grocery",
  "amount": 400,
  "lat": 12.9,
  "long": 77.6
}
```

```json
{
  "timestamp": "2026-02-06T16:10:00",
  "customer_id": "C100002",
  "merchant_id": "Amazon",
  "amount": 1200,
  "lat": 13.0,
  "long": 77.5
}
```

---

### 🚨 FRAUD
```json
{
  "timestamp": "2026-02-06T02:30:00",
  "customer_id": "C100004",
  "merchant_id": "Crypto",
  "amount": 4500,
  "lat": 13.1,
  "long": 77.6
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

```json
{
  "timestamp": "2026-02-05T02:10:00",
  "customer_id": "C000010",
  "merchant_id": "Electronics",
  "amount": 12000,
  "lat": 41.5,
  "long": 120.2
}
```

---

## 📌 Key Design Principles

- No training-serving skew
- Feature consistency via shared logic
- ML + rule hybrid fraud detection
- Fully reproducible via Docker
- Real-time inference (<100ms)
- Enterprise-style architecture

---

## 👨‍💻 Author

Built as a real-time MLOps fraud detection system using ensemble modeling (Logistic Regression + XGBoost) with feature store and real-time inference.


📐 Logical Architecture (how data flows)
----------------------------------------
pgsql

                ┌──────────────────────────┐
                │   generate_data.py       │
                │  (synthetic transactions)│
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │  transactions.csv        │
                │  (DVC tracked dataset)   │
                └────────────┬─────────────┘
                             │
                             ▼
        ┌────────────────────────────────────┐
        │        train.py (Offline)          │
        │  - Feature Engineering             │
        │  - Logistic Regression             │
        │  - XGBoost                         │
        │  - Ensemble ready                  │
        └────────────┬─────────────┬────────┘
                     │             │
                     ▼             ▼
        ┌────────────────┐   ┌────────────────┐
        │  MLflow        │   │  Metrics        │
        │  Model Store   │   │ Precision, AUC  │
        └───────┬────────┘   └────────────────┘
                │
                ▼
     ┌───────────────────────────────┐
     │        FastAPI (app.py)       │
     │  - Loads LR + XGB models      │
     │  - Ensemble probability       │
     │  - Rule + ML hybrid logic     │
     └───────────┬───────────┬──────┘
                 │           │
                 ▼           ▼
        ┌──────────────┐   ┌─────────────────┐
        │   Redis      │   │   Client / User │
        │ FeatureStore │   │   / Bank App    │
        └──────────────┘   └─────────────────┘
