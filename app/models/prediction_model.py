from fastapi import HTTPException
import numpy as np
import pandas as pd
from ..utils.model_loader import ModelLoader
from .schemas import PowerConsumptionInput, PowerConsumptionOutput

class PredictionModel:
    def __init__(self):
        try:
            self.model_loader = ModelLoader()
            self.model = self.model_loader.model
            self.scaler = self.model_loader.scaler
        except Exception as e:
            raise RuntimeError(f"Failed to initialize prediction model: {str(e)}")

    def predict(self, input_data: PowerConsumptionInput) -> PowerConsumptionOutput:
        try:
            
            input_dict = input_data.dict()
            input_df = pd.DataFrame([input_dict])
            
            
            input_scaled = self.scaler.transform(input_df)
            
            
            prediction = self.model.predict(input_scaled)
            
            return PowerConsumptionOutput(
                predicted_consumption=float(prediction[0]),
                model_used=self.model.__class__.__name__
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )