from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
import os


def evaluate_model(model, X_test, y_test, model_name="model"):
    """Evaluate a single model with advanced metrics."""
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Basic metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    # 🔥 Ensure models folder exists
    os.makedirs("models", exist_ok=True)

    # 🔥 Confusion Matrix Plot
    plt.figure()
    plt.imshow(metrics["confusion_matrix"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    plt.savefig(f"models/{model_name}_confusion_matrix.png")
    plt.close()

    # 🔥 ROC + AUC (multi-class)
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)

    auc_score = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")
    metrics["auc"] = auc_score

    # Plot ROC for first class (clean visualization)
    fpr, tpr, _ = roc_curve(y_test_bin[:, 0], y_proba[:, 0])

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(f"models/{model_name}_roc_curve.png")
    plt.close()

    return metrics


def evaluate_all_models(trained_models, X_test, y_test):
    """Evaluate all trained models."""
    
    results = {}

    for name, model in trained_models.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_model(model, X_test, y_test, name)

        print(f"Accuracy: {results[name]['accuracy']:.4f}")
        print(f"Precision: {results[name]['precision']:.4f}")
        print(f"Recall: {results[name]['recall']:.4f}")
        print(f"F1-score: {results[name]['f1_score']:.4f}")

    return results