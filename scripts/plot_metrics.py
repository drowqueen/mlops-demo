# scripts/plot_metrics.py

import joblib
import pandas as pd
from utils.plotting import plot_predicted_vs_actual, plot_feature_importance

# Load test data
X_test = pd.read_csv("data/ames_test_features.csv")  # Adjust path as needed
y_test = pd.read_csv("data/ames_test_targets.csv")["SalePrice"]  # Adjust path as needed

# Load trained model
model = joblib.load("model/ames_model.pkl")

# Generate predictions
y_pred = model.predict(X_test)

# Plot results
plot_predicted_vs_actual(y_test, y_pred)
plot_feature_importance(model, X_test.columns)
