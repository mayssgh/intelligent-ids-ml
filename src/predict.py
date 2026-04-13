import joblib


def load_model(model_path):
    """Load trained model."""
    return joblib.load(model_path)


def predict(model, input_data):
    """Make prediction."""
    prediction = model.predict(input_data)
    return prediction