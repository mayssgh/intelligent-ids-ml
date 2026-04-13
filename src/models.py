from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def get_models():
    """Return a dictionary of models."""
    
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
    }

    return models


def train_models(models, X_train, y_train):
    """Train all models."""
    
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models