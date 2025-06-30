# scripts/train_model.py
import os
import sys
import logging
import warnings
from math import sqrt
import argparse
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model_utils import save_model_with_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train model with specified algorithm.")
parser.add_argument(
    "--model",
    choices=["gb", "xgb"],
    required=True,
    help="Choose model to train: 'gb', 'xgb'",
)
args = parser.parse_args()

# Configure model and search space

if args.model == "gb":
    model = GradientBoostingRegressor(
        random_state=9999, n_iter_no_change=5, validation_fraction=0.15
    )
    param_dist = {
        "n_estimators": [300, 400, 500, 600],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.07, 0.1],
        "subsample": [0.6, 0.7, 0.85],
        "min_samples_split": [3, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    model_name = "gb"

elif args.model == "xgb":
    model = XGBRegressor(
        random_state=9999,
        verbosity=0,
        objective="reg:squarederror",
    )
    param_dist = {
        "n_estimators": [200, 300, 400],  # increased
        "max_depth": [4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    model_name = "xgb"

# Load data
print("Loading cleaned data...")
df = pd.read_csv("data/ames_cleaned.csv")
X = df[
    [
        "Gr Liv Area",
        "Overall Qual",
        "Garage Cars",
        "Year Built",
        "totalSF",
        "qualitySF",
        "finishedSF",
        "TotalBath",
        "age",
        "remodeled_age",
        "has_pool",
        "has_fireplace",
        "has_garage",
        "is_new",
        "lot_ratio",
        "porch_area",
        "log_Gr Liv Area",
        "log_Lot Area",
        "log_totalSF",
        "log_finishedSF",
        "log_qualitySF",
        "log_porch_area",
    ]
]
y = df["SalePrice"]

# Split data
print("Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=9999
)
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
y_train = pd.Series(y_train, name="SalePrice")
y_test = pd.Series(y_test, name="SalePrice")

# Save splits
print("Saving train/test splits to CSV files...")
X_train.to_csv(f"data/{model_name}_train_features.csv", index=False)
X_test.to_csv(f"data/{model_name}_test_features.csv", index=False)
y_train.to_frame().to_csv(f"data/{model_name}_train_targets.csv", index=False)
y_test.to_frame().to_csv(f"data/{model_name}_test_targets.csv", index=False)

# Hyperparameter tuning
print("Starting hyperparameter tuning...")

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,  # increased from 60
    cv=5,
    verbose=2,
    random_state=9999,
    n_jobs=-1,
    scoring="neg_mean_squared_error",
)
random_search.fit(X_train, y_train)
# Save all hyperparameter search results
cv_results_df = pd.DataFrame(random_search.cv_results_)
cv_results_df.to_csv(f"model/{model_name}_search_results.csv", index=False)
best_params = random_search.best_params_
# Cross-validation evaluation with best estimator
logger.info("Performing cross-validation evaluation with best estimator...")
kf = KFold(n_splits=5, shuffle=True, random_state=9999)
rmse_scorer = make_scorer(
    lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    greater_is_better=False,
)
cv_scores = cross_val_score(
    random_search.best_estimator_, X, y, scoring=rmse_scorer, cv=kf, n_jobs=-1
)
cv_rmse_scores = -cv_scores
logger.info(f"Cross-validation RMSE scores: {cv_rmse_scores}")
logger.info(f"Mean CV RMSE: {cv_rmse_scores.mean():.2f} ± {cv_rmse_scores.std():.2f}")

# Train final model and predict
if args.model == "xgb":
    print("Training final XGBoost model using xgb.train with early stopping...")

    # Prepare DMatrix for training and validation
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Prepare parameters, exclude n_estimators as it becomes num_boost_round
    xgb_params = {
        "objective": "reg:squarederror",
        "verbosity": 1,  # light info; 0=silent
        "random_state": 9999,
    }
    # Copy best_params except n_estimators
    for k, v in best_params.items():
        if k != "n_estimators":
            xgb_params[k] = v

    num_boost_round = best_params.get("n_estimators", 200)
    early_stopping_rounds = 20
    evals = [(dtest, "eval"), (dtrain, "train")]
    # Suppress any warning during training
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        final_model = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
    # Predict on test set
    y_pred = final_model.predict(dtest)

else:
    # For  GB just use best estimator from RandomizedSearchCV
    final_model = random_search.best_estimator_
    y_pred = final_model.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE of best model: {rmse:.2f}")

# Prediction logging - save true vs predicted
os.makedirs("predictions", exist_ok=True)
pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
pred_path = f"predictions/{model_name}_test_predictions.csv"
pred_df.to_csv(pred_path, index=False)
logger.info(f"Saved test predictions to {pred_path}")

# Save model and metadata
model_path, metadata_path, version = save_model_with_metadata(
    final_model, model_name, X_train, X_test, rmse, best_params
)
print(f"Model saved to {model_path}")
print(f"Metadata saved to {metadata_path}")
print(f"Model version {version} registered in model_registry.json")
