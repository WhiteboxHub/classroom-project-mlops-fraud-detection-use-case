from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class TransactionRequest(BaseModel):
    timestamp: str = Field(..., description="ISO 8601 timestamp of transaction")
    customer_id: str = Field(..., description="Unique customer identifier")
    merchant_id: str = Field(..., description="Unique merchant identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    lat: float = Field(..., gte=-90, lte=90, description="Latitude")
    long: float = Field(..., gte=-180, lte=180, description="Longitude")

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="FRAUD or LEGIT")
    probability: float = Field(..., description="Fraud probability (0.0 to 1.0)")
    explanation: Dict[str, float] = Field(..., description="Top contributing features")
    model_version: str = Field(..., description="Model Run ID or Version used")
