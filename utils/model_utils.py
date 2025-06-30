# utils/model_utils.py
import os
import json
import re
from datetime import datetime
import joblib


def get_next_version(model_type: str, model_dir="model") -> str:
    """Return the next version string for a model based on existing files."""
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


def save_model_with_metadata(
    model, model_type, X_train, X_test, rmse, best_params, model_dir="model"
):
    """Save the model and its metadata, then update the model registry."""
    os.makedirs(model_dir, exist_ok=True)
    version = get_next_version(model_type, model_dir)
    model_path = f"{model_dir}/{model_type}_v{version}.pkl"
    metadata_path = f"{model_dir}/{model_type}_v{version}.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_type": (
            "RandomForestRegressor"
            if model_type == "rf"
            else "GradientBoostingRegressor"
        ),
        "version": version,
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_rmse": round(rmse, 2),
        "features": list(X_train.columns),
        "best_params": best_params,
        "filename": os.path.basename(model_path),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    update_model_registry(metadata, f"{model_dir}/model_registry.json")
    return model_path, metadata_path, version


def update_model_registry(metadata, registry_path):
    """Update the model registry JSON file with new metadata."""
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = []

    registry.append(metadata)

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
