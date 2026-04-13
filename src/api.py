from fastapi import FastAPI
import numpy as np
import joblib

app = FastAPI()

# Load model once
model = joblib.load("models/random_forest.pkl")


@app.get("/")
def home():
    return {"message": "Intelligent IDS API is running"}


@app.post("/predict")
def predict(data: list):
    """
    Input: list of features
    Output: prediction
    """
    input_array = np.array(data).reshape(1, -1)
    prediction = model.predict(input_array)

    return {"prediction": int(prediction[0])}