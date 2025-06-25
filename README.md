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

## Next Steps

- Hyperparameter tuning to improve model accuracy
- Add cross-validation for more robust evaluation
- Explore additional feature engineering
- Deploy the trained model as a REST API using FastAPI

