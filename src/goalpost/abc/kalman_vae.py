"""Abstract base classes for the Kalman VAE components.

These define the interface that sport-specific implementations must follow.
The architecture is always:
    Encoder: transition_probs -> y_hat
    Kalman:  z_prev + y_hat    -> z_new
    Decoder: z                 -> reconstructed transition_probs
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch
import torch.nn as nn


class Encoder(ABC, nn.Module):
    """Encode transition probabilities into an observation vector y_hat."""

    @abstractmethod
    def forward(self, transition_probs: torch.Tensor) -> torch.Tensor:
        """Input: [batch, input_dim] or [input_dim]. Output: y_hat."""
        pass


class KalmanFilter(ABC, nn.Module):
    """Update team latent z_t using encoded observation y_hat."""

    @abstractmethod
    def predict(self, z_prev: torch.Tensor) -> torch.Tensor:
        """Predict step: z_t|t-1 from z_prev."""
        pass

    @abstractmethod
    def update(self, z_pred: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Update step: incorporate observation y_hat."""
        pass

    @abstractmethod
    def forward(self, z_prev: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Predict then update."""
        pass


class Decoder(ABC, nn.Module):
    """Decode team latent into transition probabilities."""

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Input: [batch, z_dim] or [z_dim]. Output: logits (not softmaxed)."""
        pass


class MatchupPredictor(ABC, nn.Module):
    """Mix two team latents and decode into predicted matchup transitions."""

    @abstractmethod
    def forward(self, z_home: torch.Tensor, z_away: torch.Tensor) -> torch.Tensor:
        """Input: two team latents. Output: predicted transition probabilities."""
        pass


class Trainer(ABC):
    """Train encoder + kalman + decoder on team game histories."""

    @abstractmethod
    def fit(self, team_games: Dict[str, List[torch.Tensor]], n_epochs: int = 10) -> List[float]:
        """Train on {team_id: [game_transition_tensors]}. Returns epoch losses."""
        pass

    @abstractmethod
    def get_team_latent(self, team_id: str) -> torch.Tensor:
        """Get current latent for a team."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass
