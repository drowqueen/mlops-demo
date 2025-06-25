# Ames Housing Price Prediction

This project demonstrates a machine learning pipeline to predict house prices using the Ames Housing dataset. The model is trained using scikit-learn's RandomForestRegressor and includes data cleaning, feature engineering, model training, and evaluation. `scripts/clean_data.py` generates a number of engineered features derived from the base features and saves them in new columns.

## Getting Started

### Prerequisites

- Python 3.11 (recommended)
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

 If you placed the dataset at the default path (`data/AmesHousing.csv`), just run:

    ```bash
    python scripts/clean_data.py
    ```

If your dataset is in a different location or named differently, specify it manually:

    ```bash
    python scripts/clean_data.py --input path/to/AmesHousing.csv --output path/to/cleaned.csv
    ```

2. **Train the model:**

    ```bash
    python scripts/train_model.py
    ```

This script will:
- Load the cleaned dataset (`data/ames_cleaned.csv`)
- Train a RandomForestRegressor
- Save the trained model to `model/ames_model.pkl`
- Evaluate and print the RMSE on the test set

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
    python scripts/plot_metrics.py
    ```

   This script loads test data and the trained model, then generates the plots.

2. Ensure test split CSVs (`ames_test_features.csv`, `ames_test_targets.csv`) exist in `data/`.

### Utility Functions

Reusable plotting functions are in `utils/plotting.py`


## Next Steps

- Hyperparameter tuning to improve model accuracy
- Add cross-validation for more robust evaluation
- Explore additional feature engineering
- Deploy the trained model as a REST API using FastAPI

