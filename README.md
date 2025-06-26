# Ames Housing Price Prediction

This project demonstrates a machine learning pipeline to predict house prices using the Ames Housing dataset. The models are trained using scikit-learn's RandomForestRegressor and GradientBoostingRegressor. Scripts include functions for data cleaning, feature engineering, model training and evaluation. `scripts/clean_data.py` generates a number of engineered features derived from the base features and saves them in new columns, deletes the rows lacking critical data and handles boolean conversions.

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

### Usage

1. **Clean the raw data:**
Script location is `scripts/clean_data.py` This script will:

- Generate engineered features for predictive model training
- Delete the rows lacking critical data
- Handle boolean conversions

If you placed the dataset at the default path (`data/AmesHousing.csv`), just run:

```bash
python scripts/clean_data.py
```

If your dataset is in a different location or named differently, specify it as cli arguments:

```bash
python scripts/clean_data.py --input path/to/AmesHousing.csv --output path/to/cleaned.csv
```

2. **Train the models:**

Script location is `scripts/train_model.py` This script will:

- Load the cleaned dataset (`data/ames_cleaned.csv`)
- Split the tedt data and trainiung data, saving them under `data/`
- Train a RandomForestRegressor or GradientBoostingRegressor
- Save the trained model to `model/gb_model_best.pkl` or `model/rf_model_best.pkl`
- Evaluate and print the RMSE on the test set


```bash
mkdir -p model
python scripts/train_model.py --model rf (RandomForest model)
```
or
```bash
python scripts/train_model.py --model gb (GradientBoosting model)
```

### Model Evaluation

- Root Mean Squared Error (RMSE) is used as the evaluation metric.
- Lower RMSE values indicate better model performance.

## Plotting Model Metrics

Visualize model performance with:

- **Predicted vs Actual Scatter Plot:**  
Compares predicted prices to actual sale prices. Points near the y=x line indicate good predictions; outliers show errors.

- **Feature Importance Plot:**  
Displays which features most influence the model's predictions.

### Usage

1. Run the plotting script:

```bash
python scripts/plot_metrics.py --model rf (RandomForest model)
```
or
```bash
python scripts/plot_metrics.py --model gb (GradientBoosting model)
```

  This script loads test data and the trained model, then generates the plots.

2. Ensure test split CSVs (`ames_test_features.csv`, `ames_test_targets.csv`) exist in `data/`.

## Running the FastAPI Model Serving App

### Prerequisites
- Python 3.10+
- All dependencies installed (`pip install -r requirements.txt`)
- Trained models saved in the `model/` directory (e.g., `rf_model_best.pkl`, `gb_model_best.pkl`)

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

### Utility Functions

Reusable plotting functions are in `utils/plotting.py`

## Next Steps

- Hyperparameter tuning to improve model accuracy
- Add model versioning metadata
- Add prediction logging
- Add cross-validation for more robust evaluation
- Dockerize the FastAPI app

