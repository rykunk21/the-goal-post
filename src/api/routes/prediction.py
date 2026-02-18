"""Prediction endpoint - stubs for TDD."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class PredictionRequest(BaseModel):
    home_team_id: str
    away_team_id: str
    is_neutral_site: bool = False
    iterations: int = 10000


class PredictionResponse(BaseModel):
    home_win_prob: float
    away_win_prob: float
    home_expected_score: float
    away_expected_score: float
    spread: str
    total: float
    model_version: str


@router.post("", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate game prediction.
    
    TODO: Implement prediction pipeline:
    1. Load team latent representations from database
    2. Run transition NN to get probabilities
    3. Run MCMC simulation
    4. Return predictions
    """
    # Stub implementation - will be replaced with actual prediction logic
    raise HTTPException(status_code=501, detail="Prediction not yet implemented")
