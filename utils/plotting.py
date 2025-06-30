# utils/plotting.py

import matplotlib.pyplot as plt
import numpy as np


def plot_predicted_vs_actual(y_true, y_pred, title="Predicted vs Actual Prices"):
    """
    Creates a scatter plot of predicted vs actual values.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        title (str): Plot title.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()


def plot_feature_importance(model, feature_names, title="Feature Importances"):
    """
    Creates a bar plot of feature importances from a fitted model.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names (list or array-like): List of feature names.
        title (str): Plot title.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(
        range(len(importances)), [feature_names[i] for i in indices], rotation=90
    )
    plt.ylabel("Importance")
    plt.tight_layout()
