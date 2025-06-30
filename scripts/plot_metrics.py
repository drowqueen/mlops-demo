"""Visualize predictions and feature importances for a trained regression model."""

import os
import sys
import argparse
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path for utils imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.plotting import (
    plot_predicted_vs_actual,
    plot_feature_importance,
)  # noqa: E402


def main(model_prefix: str):
    """
    Plot predictions and feature importances for a trained model.

    Args:
        model_prefix (str): Prefix like 'rf' or 'gb' used in file naming.
    """
    x_test_path = f"data/{model_prefix}_test_features.csv"
    y_test_path = f"data/{model_prefix}_test_targets.csv"
    model_path = f"model/{model_prefix}_model_best.pkl"

    print(f"Loading test features from {x_test_path}...")
    x_test = pd.read_csv(x_test_path)

    print(f"Loading test targets from {y_test_path}...")
    y_test = pd.read_csv(y_test_path)["SalePrice"]

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print("Generating predictions...")
    y_pred = model.predict(x_test)

    print("Plotting predicted vs actual...")
    plot_predicted_vs_actual(y_test, y_pred)

    print("Plotting feature importance...")
    plot_feature_importance(model, x_test.columns)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics for a trained model")
    parser.add_argument(
        "--model",
        choices=["rf", "gb"],
        required=True,
        help="Model prefix to load appropriate model/data files (e.g., 'rf' or 'gb')",
    )
    args = parser.parse_args()
    main(args.model)
