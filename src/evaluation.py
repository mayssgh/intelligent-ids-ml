from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from sklearn.preprocessing import label_binarize


def evaluate_model(model, X_test, y_test, model_name="model"):
    """Evaluate a single model with advanced metrics."""

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "accuracy":         round(accuracy_score(y_test, y_pred) * 100, 4),
        "precision":        round(precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 4),
        "recall":           round(recall_score(y_test, y_pred, average="weighted") * 100, 4),
        "f1":               round(f1_score(y_test, y_pred, average="weighted") * 100, 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    os.makedirs("models", exist_ok=True)

    # Confusion matrix plot
    plt.figure()
    plt.imshow(metrics["confusion_matrix"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.colorbar()
    plt.savefig(f"models/{model_name}_confusion_matrix.png")
    plt.close()

    # ROC + AUC (multi-class)
    classes      = np.unique(y_test)
    y_test_bin   = label_binarize(y_test, classes=classes)
    auc_score    = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")
    metrics["auc"] = round(float(auc_score) * 100, 4)

    fpr, tpr, _ = roc_curve(y_test_bin[:, 0], y_proba[:, 0])
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve — {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(f"models/{model_name}_roc_curve.png")
    plt.close()

    print(f"  Accuracy : {metrics['accuracy']}%")
    print(f"  Precision: {metrics['precision']}%")
    print(f"  Recall   : {metrics['recall']}%")
    print(f"  F1 Score : {metrics['f1']}%")
    print(f"  AUC      : {metrics['auc']}%")

    return metrics


def evaluate_all_models(trained_models, X_test, y_test, cv_scores=None, dataset_info=None):
    """Evaluate all trained models and save metrics.json for the dashboard."""

    results = {}
    for name, model in trained_models.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_model(model, X_test, y_test, name)

    # ── Save metrics.json so the dashboard can load real values ──
    best = max(results, key=lambda n: results[n]["f1"])

    # Build confusion matrix summary for best model (binary: attack vs benign)
    cm = results[best]["confusion_matrix"]
    # Assumes class 0 = BENIGN, everything else = Attack
    # For multi-class we use row sums as a simplified binary view
    if len(cm) == 2:
        tp, fp, fn, tn = cm[1][1], cm[0][1], cm[1][0], cm[0][0]
    else:
        # Multi-class: TP = sum of diagonal except index 0, etc.
        cm_arr = np.array(cm)
        tn = int(cm_arr[0, 0])
        fp = int(cm_arr[0, 1:].sum())
        fn = int(cm_arr[1:, 0].sum())
        tp = int(cm_arr[1:, 1:].sum())

    metrics_out = {
        "models": {
            name: {
                "accuracy":  m["accuracy"],
                "precision": m["precision"],
                "recall":    m["recall"],
                "f1":        m["f1"],
                "auc":       m["auc"]
            }
            for name, m in results.items()
        },
        "best_model": best,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "cv_scores": [round(float(s)*100, 2) for s in cv_scores] if cv_scores is not None else [],
        "dataset": dataset_info or {}
    }

    with open("models/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("\n✅ metrics.json saved → models/metrics.json")

    return results