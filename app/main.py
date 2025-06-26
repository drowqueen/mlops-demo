"""FastAPI app for Ames Housing price prediction."""

from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field


app = FastAPI()

# Cache for loaded models
model_cache = {}


# Pydantic input model
class HouseFeatures(BaseModel):
    Gr_Liv_Area: float = Field(..., alias="Gr Liv Area")
    Overall_Qual: int = Field(..., alias="Overall Qual")
    Garage_Cars: int = Field(..., alias="Garage Cars")
    Year_Built: int = Field(..., alias="Year Built")
    totalSF: float
    qualitySF: float
    finishedSF: float
    TotalBath: float
    age: int
    remodeled_age: int
    has_pool: int
    has_fireplace: int
    has_garage: int
    is_new: int
    lot_ratio: float
    porch_area: float
    model: Literal["rf", "gb"]


@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.post("/predict")
def predict(features: HouseFeatures):
    model_name = features.model
    model_path = Path(f"model/{model_name}_model_best.pkl")

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    if model_name not in model_cache:
        model_cache[model_name] = joblib.load(model_path)

    model = model_cache[model_name]

    input_df = pd.DataFrame(
        [
            {
                "Gr Liv Area": features.Gr_Liv_Area,
                "Overall Qual": features.Overall_Qual,
                "Garage Cars": features.Garage_Cars,
                "Year Built": features.Year_Built,
                "totalSF": features.totalSF,
                "qualitySF": features.qualitySF,
                "finishedSF": features.finishedSF,
                "TotalBath": features.TotalBath,
                "age": features.age,
                "remodeled_age": features.remodeled_age,
                "has_pool": features.has_pool,
                "has_fireplace": features.has_fireplace,
                "has_garage": features.has_garage,
                "is_new": features.is_new,
                "lot_ratio": features.lot_ratio,
                "porch_area": features.porch_area,
            }
        ]
    )

    prediction = model.predict(input_df)[0]
    return {"predicted_price": round(prediction, 2), "model": model_name}
