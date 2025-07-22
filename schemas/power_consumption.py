from pydantic import BaseModel, Field
from typing import Optional

class PowerConsumptionInput(BaseModel):
    temperature: float = Field(..., ge=-20, le=50, description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    wind_speed: float = Field(..., ge=0, le=100, description="Wind speed in km/h")
    general_diffuse_flows: float = Field(..., ge=0, le=1, description="General diffuse flows")
    diffuse_flows: float = Field(..., ge=0, le=1, description="Diffuse flows")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0-6, where 0 is Monday)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")

class PowerConsumptionOutput(BaseModel):
    predicted_consumption: float
    model_used: str