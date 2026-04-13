from preprocessing import preprocess_pipeline
from models import get_models, train_models
from evaluation import evaluate_all_models
from explainability import explain_model

import joblib
import os

# CONFIG
FILE_PATH = "data/raw/Wednesday-workingHours.pcap_ISCX.csv"
TARGET = "Label"


def main():
    print("Starting training pipeline...")

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess_pipeline(
        FILE_PATH, TARGET
    )

    # Get models
    models = get_models()

    # Train models
    trained_models = train_models(models, X_train, y_train)

    print("\nTraining completed.")

    # Evaluate models
    results = evaluate_all_models(trained_models, X_test, y_test)

    # 🔥 Select best model
    best_model = trained_models["random_forest"]

    # 🔍 Explain model (use small sample for speed)
    explain_model(best_model, X_test[:200])

    # 💾 Save model safely
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/random_forest.pkl")
    print("✅ Model saved in models/random_forest.pkl")

    return results


if __name__ == "__main__":
    main()