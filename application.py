from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import psutil
import numpy as np
from typing import List, Optional

app = FastAPI(title="Power Consumption API")

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {mb:.2f} MB")

model = None
scaler = None

def get_model():
    global model, scaler
    if model is None or scaler is None:
        try:
            model = joblib.load('app/models/power_consumption_model_compressed.joblib')
            scaler = joblib.load('app/models/scaler.pkl')
            log_memory_usage()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    return model, scaler

class PredictionInput(BaseModel):
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    hour: int
    day_of_week: int
    is_weekend: int
    is_holiday: int

@app.get("/")
async def root():
    return {
        "message": "Power Consumption Prediction API",
        "status": "operational",
        "memory_usage": f"{psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f}MB"
    }

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        model, scaler = get_model()
        
        input_features = np.array([
            [
                input_data.temperature,
                input_data.humidity,
                input_data.pressure,
                input_data.wind_speed,
                input_data.hour,
                input_data.day_of_week,
                input_data.is_weekend,
                input_data.is_holiday
            ]
        ])
        
        scaled_features = scaler.transform(input_features)
        
        prediction = model.predict(scaled_features)
        
        prediction_value = float(prediction[0])
        
        return {
            "prediction": prediction_value,
            "units": "kWh"  
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("application:app", host="0.0.0.0", port=8000, reload=True)