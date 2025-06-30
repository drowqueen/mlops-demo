# scripts/train_model.py
from math import sqrt
import pandas as pd  # loads and handles tabular data
import joblib  # saves and loads machine learning models
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)  # machine learning models
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error  # measures the accuracy of the model
import argparse
import os, json, re  # basic file & string tools
from datetime import datetime  # get current date/time

# Parse command line arguments to choose the model to train
parser = argparse.ArgumentParser(description="Train model with specified algorithm.")
parser.add_argument(
    "--model",
    choices=["rf", "gb"],
    required=True,
    help="Choose model to train, rf: random forest gb: gradient boosting",
)
args = parser.parse_args()

# Determine model and parameters first
if args.model == "rf":
    model = RandomForestRegressor(random_state=9999)
    param_dist = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    model_name = "rf"
elif args.model == "gb":
    model = GradientBoostingRegressor(random_state=9999)
    param_dist = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "min_samples_split": [2, 5, 10],
    }
    model_name = "gb"
else:
    raise ValueError(f"Unknown model type: {args.model}")

# Load data
print("Loading cleaned data...")
df = pd.read_csv("data/ames_cleaned.csv")  # Cleaned Ames housing data
print(f"Data shape: {df.shape}")

# Select the base + engineered features and target
print("Selecting features and target...")
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
    ]
]
y = df["SalePrice"]
print(f"Feature matrix shape: {X.shape}, Target vector length: {len(y)}")

# Train/test split - splits data into training and testing sets
print("Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=9999
)

# Ensure types (convert numpy arrays/lists to DataFrames/Series)
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

y_train = pd.Series(y_train, name="SalePrice")
y_test = pd.Series(y_test, name="SalePrice")

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Save the splits to CSV files
print("Saving train/test splits to CSV files...")
X_train.to_csv(f"data/{model_name}_train_features.csv", index=False)
X_test.to_csv(f"data/{model_name}_test_features.csv", index=False)

y_train.to_frame().to_csv(f"data/{model_name}_train_targets.csv", index=False)
y_test.to_frame().to_csv(f"data/{model_name}_test_targets.csv", index=False)

print("Train/test splits saved!")

# Hyperparameter tuning with RandomizedSearchCV
print("Starting hyperparameter tuning...")

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    verbose=2,
    random_state=9999,
    n_jobs=-1,
    scoring="neg_mean_squared_error",
)

random_search.fit(X_train, y_train)

print("Best parameters found:")
print(random_search.best_params_)

best_model = random_search.best_estimator_

# Evaluate the accuracy of the best model
print("Evaluating the best model on test set...")
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
print(f"Test RMSE of best model: {rmse:.2f}")

print("Model evaluation complete!")


# === Model Versioning Utilities ===
def get_next_version(model_type: str, model_dir="model") -> str:
    """Automatically determines the next semantic version for the model"""

    existing_files = os.listdir(model_dir)
    versions = []
    pattern = re.compile(rf"{model_type}_v(\d+)\.(\d+)\.(\d+)\.pkl")

    for fname in existing_files:
        match = pattern.match(fname)
        if match:
            versions.append(tuple(map(int, match.groups())))

    if versions:
        latest = max(versions)
        next_version = (latest[0], latest[1], latest[2] + 1)
    else:
        next_version = (1, 0, 0)

    return f"{next_version[0]}.{next_version[1]}.{next_version[2]}"


# === Save versioned model and metadata ===
print("Saving versioned model and metadata...")
os.makedirs("model", exist_ok=True)
version = get_next_version(model_name)
model_path = f"model/{model_name}_v{version}.pkl"
metadata_path = f"model/{model_name}_v{version}.json"

joblib.dump(best_model, model_path)

metadata = {
    "model_type": (
        "RandomForestRegressor" if model_name == "rf" else "GradientBoostingRegressor"
    ),
    "version": version,
    "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "test_rmse": round(rmse, 2),
    "features": list(X.columns),
    "best_params": random_search.best_params_,
    "filename": os.path.basename(model_path),
}

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Model saved to {model_path}")
print(f"Metadata saved to {metadata_path}")

# === Update model registry ===
registry_path = "model/model_registry.json"

if os.path.exists(registry_path):
    with open(registry_path, "r") as f:
        registry = json.load(f)
else:
    registry = []

registry.append(metadata)

with open(registry_path, "w") as f:
    json.dump(registry, f, indent=2)

print(f"Model version {version} registered in model_registry.json")
