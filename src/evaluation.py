from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(model, X_test, y_test):
    """Evaluate a single model."""
    
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return metrics


def evaluate_all_models(trained_models, X_test, y_test):
    """Evaluate all trained models."""
    
    results = {}

    for name, model in trained_models.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_model(model, X_test, y_test)

        print(f"Accuracy: {results[name]['accuracy']:.4f}")
        print(f"Precision: {results[name]['precision']:.4f}")
        print(f"Recall: {results[name]['recall']:.4f}")
        print(f"F1-score: {results[name]['f1_score']:.4f}")

    return results