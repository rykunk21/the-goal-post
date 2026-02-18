"""NCAAB Predictor API.

FastAPI application for NCAAB basketball predictions using VAE + NN + MCMC.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import routes

app = FastAPI(
    title="NCAAB Predictor",
    description="ML-powered NCAAB basketball prediction system",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes.health.router, prefix="/health", tags=["Health"])
app.include_router(routes.prediction.router, prefix="/predict", tags=["Prediction"])
app.include_router(routes.training.router, prefix="/train", tags=["Training"])
app.include_router(routes.model.router, prefix="/model", tags=["Model"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "NCAAB Predictor",
        "version": "0.1.0",
        "docs": "/docs"
    }
