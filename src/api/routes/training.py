"""Training endpoints - stubs for TDD."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class VAETrainingRequest(BaseModel):
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    latent_dim: int = 16


class NNTrainingRequest(BaseModel):
    epochs: int = 50
    learning_rate: float = 0.0001
    batch_size: int = 32


class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str


@router.post("/vae", response_model=TrainingResponse)
async def train_vae(request: VAETrainingRequest):
    """Train VAE component.
    
    TODO: Implement VAE training:
    1. Load historical game data
    2. Train VAE with reconstruction + KL + InfoNCE loss
    3. Save model weights
    4. Return job status
    """
    raise HTTPException(status_code=501, detail="VAE training not yet implemented")


@router.post("/nn", response_model=TrainingResponse)
async def train_nn(request: NNTrainingRequest):
    """Train Transition Probability NN.
    
    TODO: Implement NN training:
    1. Load team latent representations
    2. Load ground truth transition probabilities
    3. Train NN with cross-entropy loss
    4. Save model weights
    """
    raise HTTPException(status_code=501, detail="NN training not yet implemented")
