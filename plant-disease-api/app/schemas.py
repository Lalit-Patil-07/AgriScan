from pydantic import BaseModel, Field
from typing import Optional, List

class DiseaseInfo(BaseModel):
    """Detailed information about the detected disease."""
    description: str = Field(..., example="A detailed description of the disease.")
    symptoms: List[str] = Field(..., example=["Symptom 1", "Symptom 2"])
    prevention: List[str] = Field(..., example=["Tip 1", "Tip 2"])

class PredictionResponse(BaseModel):
    """The final JSON response sent to the client."""
    disease_name: str = Field(..., example="Apple Scab")
    confidence: float = Field(..., gt=0, le=1, example=0.9852)
    details: DiseaseInfo