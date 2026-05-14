from preprocessing import preprocess_pipeline
from models import get_models, train_models
from evaluation import evaluate_all_models
from explainability import explain_model
from sklearn.model_selection import cross_val_score
from collections import Counter
from imblearn.over_sampling import SMOTE
import joblib
import os

FILE_PATH = "data/raw/Wednesday-workingHours.pcap_ISCX.csv"
TARGET    = "Label"


def main():
    print("Starting training pipeline...")

    # Preprocess
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess_pipeline(FILE_PATH, TARGET)

    print("\nClass distribution BEFORE balancing:")
    print(Counter(y_train))

    print("\nApplying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("\nClass distribution AFTER balancing:")
    after = Counter(y_train)
    print(after)

    # Train
    models        = get_models()
    trained_models = train_models(models, X_train, y_train)

    # Cross-validation on Random Forest
    print("\nPerforming Cross-Validation on Random Forest...")
    rf_model  = trained_models["random_forest"]
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    print("CV scores:", cv_scores)
    print("Mean CV :", cv_scores.mean())

    os.makedirs("models", exist_ok=True)

    # SHAP explainability
    explain_model(rf_model, X_test[:200])

    # Evaluate — now also saves models/metrics.json for the dashboard
    dataset_info = {
        "total":      int(len(X_train) + len(X_test)),
        "train_size": int(len(X_train)),
        "test_size":  int(len(X_test)),
        "after_smote":int(len(y_train)),
        "classes":    {str(k): int(v) for k, v in Counter(y_test).items()}
    }

    results = evaluate_all_models(
        trained_models, X_test, y_test,
        cv_scores=cv_scores,
        dataset_info=dataset_info
    )

    # Save model
    joblib.dump(rf_model, "models/random_forest.pkl")
    print("✅ Model saved → models/random_forest.pkl")

    return results


if __name__ == "__main__":
    main()