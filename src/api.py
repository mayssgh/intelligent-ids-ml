from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import os

app = FastAPI(
    title="Intelligent IDS API",
    description="Real-time intrusion detection using ML",
    version="1.0"
)

# ---------- Load Model ----------
MODEL_PATH = "models/random_forest.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded")
else:
    model = None
    print("⚠️ Model not found")


# ---------- Input Schema ----------
class InputData(BaseModel):
    features: List[float]


# ---------- Routes ----------
@app.get("/")
def home():
    return {"message": "Intelligent IDS API is running"}


@app.post("/predict")
def predict(data: InputData):

    EXPECTED_FEATURES = model.n_features_in_ if model else len(data.features)

    features = np.array(data.features)

    # Auto-fix size for demo
    if len(features) < EXPECTED_FEATURES:
        features = np.pad(features, (0, EXPECTED_FEATURES - len(features)))
    elif len(features) > EXPECTED_FEATURES:
        features = features[:EXPECTED_FEATURES]

    features = features.reshape(1, -1)

    if model is not None:
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        confidence = float(max(probabilities))

        is_attack = prediction != "BENIGN"
    else:
        # fallback demo mode
        score = float(np.mean(np.abs(features)))
        is_attack = score > 0.3
        confidence = 0.9

    result = str(prediction)

    return {
        "prediction": result,
        "is_attack": is_attack,
        "confidence": round(confidence, 4)
    }