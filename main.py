from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import os

app = FastAPI(
    title="Intelligent IDS API",
    description="Real-time intrusion detection using Machine Learning",
    version="1.0.0"
)

MODEL_PATH = "models/random_forest.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
else:
    model = None
    print("Model not found - running in demo mode")


class InputData(BaseModel):
    features: List[float]


@app.get("/")
def home():
    return {
        "message": "Intelligent IDS API is running",
        "status": "ok"
    }


@app.get("/health")
def health():
    return {
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)

    if model is not None:
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        is_attack = prediction != 0
    else:
        score = float(np.mean(np.abs(features)))
        is_attack = score > 0.3
        confidence = 0.9

    result = "Attack Detected" if is_attack else "Benign Traffic"

    return {
        "prediction": result,
        "confidence": round(confidence, 4)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)