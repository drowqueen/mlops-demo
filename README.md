# Ames Housing Price Prediction

This project demonstrates a machine learning pipeline to predict house prices using the Ames Housing dataset. The models are trained using scikit-learn's RandomForestRegressor and GradientBoostingRegressor. Scripts include functions for data cleaning, feature engineering, model training and evaluation. `scripts/clean_data.py` generates a number of engineered features derived from the base features and saves them in new columns, deletes the rows lacking critical data and handles boolean conversions.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Preparing the data](#preparing-data)
- [Training the models](#training-models)
- [Plotting Model Metrics](#plotting-model-metrics)
- [Running the FastAPI Model Serving App](#running-the-fastapi-model-serving-app)
- [Feature Engineering Functions in `clean_data.py`](#feature-engineering-functions-in-clean_datapy)
- [Hyperparameter Sets in `train_model.py`](#hyperparameter-sets-in-train_modelpy)
- [Main Functions in `train_model.py`](#main-functions-in-train_modelpy)
- [Planned features](#planned-features)

## Getting Started

### Prerequisites

- Python 3.10+ 
- pip

### Setup

1. Clone the repository:

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download Ames housing data set from `https://www.kaggle.com/datasets/prevek18/ames-housing-dataset`

5. Move the downloaded `AmesHousing.csv` into the project's `data/` folder:

```bash
mkdir -p data
mv ~/Downloads/AmesHousing.csv data/
```

## Preparing the data

Script location is `scripts/clean_data.py` This script will:

- Generate engineered features for predictive model training
- Delete the rows lacking critical data
- Handle boolean conversions

### Usage

If you placed the dataset at the default path (`data/AmesHousing.csv`), just run:

```bash
python scripts/clean_data.py
```

If your dataset is in a different location or named differently, specify it as cli arguments:

```bash
python scripts/clean_data.py --input path/to/AmesHousing.csv --output path/to/cleaned.csv
```

## Training the models

Script location is `scripts/train_model.py` This script will:

- Load the cleaned dataset (`data/ames_cleaned.csv`)
- Split the tedt data and trainiung data, saving them under `data/`
- Train a RandomForestRegressor or GradientBoostingRegressor
- Save the trained model to `model/gb_model_best.pkl` or `model/rf_model_best.pkl`
- Evaluate and print the RMSE on the test set

### Usage

```bash
mkdir -p model
python scripts/train_model.py --model rf (RandomForest model)
```
or
```bash
python scripts/train_model.py --model gb (GradientBoosting model)
```

## Plotting Model Metrics

 `plot_metrics.py`

This script generates visualizations to help evaluate the performance of a trained model on the test dataset. This script will:

- Load the test feature data and target values from CSV files based on the chosen model prefix (`rf` for Random Forest or `gb` for Gradient Boosting).
- Load the trained model (`.pkl` file) from the `model/` directory.
- Use the model to generate predictions on the test features.
- Plot two key visualizations:
  - **Predicted vs Actual Scatter Plot:** shows how closely predicted prices match the actual sale prices.
  - **Feature Importance Plot:** displays which features most influenced the model’s predictions.

### Usage

Run the script with the `--model` argument specifying which model to use:

    ```bash
    python scripts/plot_metrics.py --model rf
    ```
or

    ```bash
    python scripts/plot_metrics.py --model gb
    ```

Make sure the following files exist before running:

- Test features CSV: `data/rf_test_features.csv` or `data/gb_test_features.csv`
- Test targets CSV: `data/rf_test_targets.csv` or `data/gb_test_targets.csv`
- Trained model pickle: `model/rf_model_best.pkl` or `model/gb_model_best.pkl`

This tool helps visually assess model accuracy and interpret feature contributions.

## Running the FastAPI Model Serving App

### Prerequisites
- Python 3.10+
- All dependencies installed (`pip install -r requirements.txt`)
- Trained models saved in the `model/` directory (e.g., `rf_model_best.pkl`, `gb_model_best.pkl`)

### Usage

1. Start the FastAPI server

```bash
uvicorn app.main:app --reload
```

  You should see output similar to:

  ```
  INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
  ```

2. Test the health check endpoint
  Use curl or a browser to check the `/ping` endpoint:

  ```bash
  curl http://127.0.0.1:8000/ping
  ```

  Expected response:

  ```json
  {"ping":"pong"}
  ```

3. Test the prediction endpoint
  Send a POST request with JSON body containing features and a `model` key (`"rf"` or `"gb"`):
  Example command:

  ```bash
  curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "Gr Liv Area": 1500,
      "Overall Qual": 7,
      "Garage Cars": 2,
      "Year Built": 2005,
      "totalSF": 2500,
      "qualitySF": 2400,
      "finishedSF": 2300,
      "TotalBath": 2.5,
      "age": 10,
      "remodeled_age": 5,
      "has_pool": 0,
      "has_fireplace": 1,
      "has_garage": 1,
      "is_new": 0,
      "lot_ratio": 0.5,
      "porch_area": 100,
      "model": "rf"
    }'
  ```

  Expected response example:

  ```json
  {
    "predicted_price": 183631.34,
    "model": "rf"
  }
  ```

  To use the Gradient Boosting model, change `"model": "gb"` in the JSON.

## Feature Engineering Functions in `clean_data.py`

This section explains the key feature engineering functions from `scripts/clean_data.py` that transform raw data into features useful for model training.

| Function | Description |
| -------- | ----------- |
| `add_total_sf` | Calculates total square footage as the sum of 1st floor, 2nd floor, and basement square footage. |
| `add_finished_sf` | Calculates finished square footage by subtracting unfinished basement space from total SF. |
| `add_high_quality_sf` | Calculates high-quality finished SF by subtracting low-quality finished SF from total SF. |
| `add_total_bath` | Computes total bathrooms as full baths plus half baths weighted by 0.5. |
| `add_age` | Calculates house age at sale as difference between year sold and year built. |
| `add_remodeled_age` | Calculates years since last remodel at sale time. |
| `add_has_pool` | Binary feature: 1 if pool area > 0, else 0. |
| `add_has_fireplace` | Binary feature: 1 if fireplaces > 0, else 0. |
| `add_has_garage` | Binary feature: 1 if garage cars > 0, else 0. |
| `add_is_new` | Binary feature: 1 if house built within 5 years of sale, else 0. |
| `add_lot_ratio` | Ratio of lot area to above-ground living area plus one (to avoid division by zero). |
| `add_porch_area` | Sum of various porch area types (open, enclosed, 3-season, screen). |

These engineered features capture important property characteristics beyond raw input columns, improving model predictive power.

---

## Hyperparameter Sets in `train_model.py`

The hyperparameters control how the models are trained and tuned during `RandomizedSearchCV`.

| Parameter | RandomForestRegressor (`rf`) | GradientBoostingRegressor (`gb`) | Description |
| --------- | ---------------------------- | -------------------------------- | ----------- |
| `n_estimators` | [100, 200, 300, 400, 500] | [100, 200, 300, 400, 500] | Number of trees in the ensemble; more trees often improve performance but increase training time. |
| `max_depth` | [None, 10, 20, 30, 40, 50] | [3, 5, 7, 9] | Maximum depth of each tree; controls model complexity and overfitting risk. |
| `min_samples_split` | [2, 5, 10] | [2, 5, 10] | Minimum number of samples required to split a node; higher values prevent overfitting. |
| `min_samples_leaf` | [1, 2, 4] | *Not used* | Minimum samples per leaf node (only in RF); affects model smoothness. |
| `max_features` | ["sqrt", "log2", None] | *Not used* | Number of features to consider when looking for best split (only in RF); controls randomness. |
| `learning_rate` | *Not used* | [0.01, 0.05, 0.1, 0.2] | Step size shrinkage for updating trees; lower values require more trees but can improve accuracy. |
| `subsample` | *Not used* | [0.6, 0.8, 1.0] | Fraction of samples used per tree (stochastic gradient boosting); can reduce overfitting. |

The `RandomizedSearchCV` tries random combinations from these ranges to find the best-performing model.

---

## Main Functions in `train_model.py`

This script automates the workflow of training and evaluating either a Random Forest or Gradient Boosting model.

- **Argument parsing:** User selects model type (`rf` or `gb`) via command line.
- **Data loading:** Loads cleaned data CSV.
- **Feature/target selection:** Selects predefined features and target variable (`SalePrice`).
- **Train/test split:** Splits data randomly into training (80%) and testing (20%) sets.
- **Hyperparameter tuning:** Uses `RandomizedSearchCV` with defined parameter grid to search best hyperparameters via cross-validation.
- **Model training:** Fits the best model on training data.
- **Model evaluation:** Predicts on test set and computes RMSE (root mean squared error) to assess accuracy.
- **Versioned saving:** Saves the best model with an incremented semantic version number plus metadata JSON file.
- **Model registry update:** Adds metadata entry to a JSON registry file for tracking all saved model versions.

This modular approach ensures reproducible training and easy tracking of model improvements over time.


## Planned features

- Hyperparameter tuning to improve model accuracy
- Add prediction logging
- Add cross-validation for more robust evaluation
- Dockerize the FastAPI app
