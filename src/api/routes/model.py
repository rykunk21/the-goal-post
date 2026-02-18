"""Model management endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ModelStatusResponse(BaseModel):
    vae_trained: bool
    nn_trained: bool
    training_data_games: int
    model_version: str


@router.get("/status", response_model=ModelStatusResponse)
async def model_status():
    """Get current model status."""
    return ModelStatusResponse(
        vae_trained=False,
        nn_trained=False,
        training_data_games=0,
        model_version="0.0.0"
    )


@router.post("/update")
async def model_update():
    """Online learning update from recent game."""
    # TODO: Implement online learning update
    pass
