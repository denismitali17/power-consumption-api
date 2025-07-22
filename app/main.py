from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
from schemas.power_consumption import PowerConsumptionInput, PowerConsumptionOutput

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.path.join(BASE_DIR, 'app', 'models', 'optimized_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'app', 'models', 'scaler.pkl')

app = FastAPI(
    title="Power Consumption API",
    description="API for predicting power consumption in Zone 3",
    version="1.0.0",
    docs_url="/docs", 
    redoc_url="/redoc"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Loading scaler from: {SCALER_PATH}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir(os.path.dirname(MODEL_PATH))}")
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully!")
except Exception as e:
    error_msg = f"""
    Error loading model or scaler: {str(e)}
    Current directory: {os.getcwd()}
    Model path: {MODEL_PATH}
    Scaler path: {SCALER_PATH}
    Directory exists: {os.path.exists(os.path.dirname(MODEL_PATH))}
    """
    print(error_msg)
    raise RuntimeError(error_msg)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Power Consumption Prediction API",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PowerConsumptionOutput)
async def predict_power_consumption(input_data: PowerConsumptionInput):
    """
    Predict power consumption based on input features.
    
    - **temperature**: Temperature in Celsius (-20 to 50)
    - **humidity**: Humidity percentage (0-100)
    - **wind_speed**: Wind speed in km/h (0-100)
    - **general_diffuse_flows**: General diffuse flows (0-1)
    - **diffuse_flows**: Diffuse flows (0-1)
    - **hour**: Hour of day (0-23)
    - **day_of_week**: Day of week (0-6, where 0 is Monday)
    - **month**: Month (1-12)
    """
    try:
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])
        
        feature_order = [
            'temperature', 'humidity', 'wind_speed', 
            'general_diffuse_flows', 'diffuse_flows',
            'hour', 'day_of_week', 'month'
        ]
        input_df = input_df[feature_order]
        
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)
        
        return {
            "predicted_consumption": float(prediction[0]),
            "model_used": model.__class__.__name__
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)