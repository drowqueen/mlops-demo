# scripts/train_model.py
import pandas as pd  # loads and handles tabular data
import joblib  # saves and loads machine learning models
from sklearn.ensemble import RandomForestRegressor  # machine learning model
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error  # measures the accuracy of the model
from math import sqrt

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
import pandas as pd

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

y_train = pd.Series(y_train, name="SalePrice")
y_test = pd.Series(y_test, name="SalePrice")

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Save the splits to CSV files
print("Saving train/test splits to CSV files...")
X_train.to_csv("data/ames_train_features.csv", index=False)
X_test.to_csv("data/ames_test_features.csv", index=False)

y_train.to_frame().to_csv("data/ames_train_targets.csv", index=False)
y_test.to_frame().to_csv("data/ames_test_targets.csv", index=False)

print("Train/test splits saved!")


# Hyperparameter tuning with RandomizedSearchCV
print("Starting hyperparameter tuning...")
param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],  # removed 'auto'
}

rf = RandomForestRegressor(random_state=9999)

random_search = RandomizedSearchCV(
    estimator=rf,
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

# Saves the best model
print("Saving the best model...")
joblib.dump(best_model, "model/ames_model_best.pkl")
print("Best model saved!")

# Evaluates the accuracy of the best model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
print(f"Test RMSE of best model: {rmse}")
print("Model evaluation complete!")
