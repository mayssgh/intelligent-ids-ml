import shap
import numpy as np


def explain_model(model, X_sample):
    """
    Generate SHAP explanations for a model.
    
    Parameters:
        model: trained model
        X_sample: sample of data (not full dataset)
    """

    print("Generating SHAP explanations...")

    # Use a small sample (important for performance)
    if X_sample.shape[0] > 100:
        X_sample = X_sample[:100]

    # Create explainer
    explainer = shap.Explainer(model, X_sample)

    # Compute SHAP values
    shap_values = explainer(X_sample)

    # Summary plot
    shap.summary_plot(shap_values, X_sample)

    print("SHAP explanation completed.")