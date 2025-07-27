from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import List, Optional

app = FastAPI(
    title="Power Consumption API",
    description="API for predicting power consumption in Tétouan, Morocco",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
try:
    model = joblib.load('app/models/optimized_model.joblib')
    scaler = joblib.load('app/models/scaler.pkl')
    print(" Model and scaler loaded successfully")
except Exception as e:
    print(f" Error loading model or scaler: {str(e)}")
    model = None
    scaler = None

class PredictionInput(BaseModel):
    Temperature: float
    Humidity: float
    WindSpeed: float
    GeneralDiffuseFlows: float
    DiffuseFlows: float
    Hour: int
    DayOfWeek: int  # 0-6 (Monday=0)
    Month: int      # 1-12

@app.get("/")
def read_root():
    return {
        "message": "Power Consumption Prediction API",
        "status": "running",
        "docs": "/docs"
    }

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    
    try:
        # Convert input to 2D array for prediction
        input_array = np.array([[
            input_data.Temperature,
            input_data.Humidity,
            input_data.WindSpeed,
            input_data.GeneralDiffuseFlows,
            input_data.DiffuseFlows,
            input_data.Hour,
            input_data.DayOfWeek,
            input_data.Month
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        return {
            "prediction": float(prediction[0]),
            "units": "kW",
            "model": "RandomForest"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)