from pydantic import BaseModel, Field, confloat, conint
from typing import Optional

class PowerConsumptionInput(BaseModel):
    Temperature: confloat(ge=-20.0, le=50.0) = Field(..., description="Temperature in Celsius (-20 to 50)")
    Humidity: confloat(ge=0.0, le=100.0) = Field(..., description="Humidity percentage (0-100)")
    WindSpeed: confloat(ge=0.0, le=100.0) = Field(..., description="Wind speed in km/h (0-100)")
    GeneralDiffuseFlows: confloat(ge=0.0, le=1.0) = Field(..., description="General diffuse flows (0-1)")
    DiffuseFlows: confloat(ge=0.0, le=1.0) = Field(..., description="Diffuse flows (0-1)")
    Hour: conint(ge=0, le=23) = Field(..., description="Hour of day (0-23)")
    DayOfWeek: conint(ge=0, le=6) = Field(..., description="Day of week (0-6, where 0 is Monday)")
    Month: conint(ge=1, le=12) = Field(..., description="Month (1-12)")

class PowerConsumptionOutput(BaseModel):
    predicted_consumption: float = Field(..., description="Predicted power consumption in kW")
    model_used: str = Field(..., description="Name of the model used for prediction")