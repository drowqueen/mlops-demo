# scripts/plot_metrics.py

import argparse
import joblib
import pandas as pd
from utils.plotting import plot_predicted_vs_actual, plot_feature_importance


def main(model_name: str):
    # Load test data
    X_test_path = f"data/{model_name}_test_features.csv"
    y_test_path = f"data/{model_name}_test_targets.csv"
    model_path = f"model/{model_name}_model_best.pkl"

    print(f"Loading test features from {X_test_path}...")
    X_test = pd.read_csv(X_test_path)
    print(f"Loading test targets from {y_test_path}...")
    y_test = pd.read_csv(y_test_path)["SalePrice"]

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print("Generating predictions...")
    y_pred = model.predict(X_test)

    print("Plotting predicted vs actual...")
    plot_predicted_vs_actual(y_test, y_pred)

    print("Plotting feature importance...")
    plot_feature_importance(model, X_test.columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics for a trained model")
    parser.add_argument(
        "--model",
        choices=["rf", "gb"],
        required=True,
        help="Model name prefix to load data and model files (e.g., 'rf' or 'gb')",
    )
    args = parser.parse_args()
    main(args.model)
