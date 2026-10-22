from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PredictionRequest(BaseModel):
    vessel_count: float = Field(..., ge=0, le=100, description="Current number of active vessels in port")
    low_speed_vessels: float = Field(..., ge=0, le=100, description="Number of stationary or slow vessels (<=0.5 kt)")
    avg_sog: float = Field(default=2.5, ge=0.0, le=50.0, description="Average Speed Over Ground across active fleet")
    cargo_vessels: int = Field(default=4, ge=0, le=50, description="Count of active commercial cargo vessels")
    tug_vessels: int = Field(default=5, ge=0, le=50, description="Count of harbor tugs assisting traffic")
    military_vessels: int = Field(default=3, ge=0, le=50, description="Count of naval / government craft")
    hour: Optional[int] = Field(default=14, ge=0, le=23, description="Hour of day (0-23)")
    dayofweek: Optional[int] = Field(default=2, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")

    vessel_count_lag_1h: Optional[float] = None
    vessel_count_lag_2h: Optional[float] = None
    vessel_count_rolling_mean_6h: Optional[float] = None
    vessel_count_rolling_mean_24h: Optional[float] = None

class PredictionExplanation(BaseModel):
    feature: str
    impact: str
    description: str

class PredictionResponse(BaseModel):
    risk_class: int
    risk_label: str
    risk_probability: float
    confidence_level: str
    threshold: float
    timestamp: str
    explanations: List[PredictionExplanation]
    inputs_used: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    mongodb_connected: bool
    model_loaded: bool
    version: str
