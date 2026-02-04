from pydantic import BaseModel

class TransactionRequest(BaseModel):
    timestamp: str
    customer_id: str
    merchant_id: str
    amount: float
    lat: float
    long: float

class Explanation(BaseModel):
    model_confidence: float
    decision_reason: str

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    explanation: Explanation
    model_version: str
