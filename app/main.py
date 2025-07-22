from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from schemas.power_consumption import PowerConsumptionInput, PowerConsumptionOutput

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
    model = joblib.load('app/models/power_consumption_model.pkl')
    scaler = joblib.load('app/models/scaler.pkl')
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {str(e)}")

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
    uvicorn.run(app, host="0.0.0.0", port=8000)