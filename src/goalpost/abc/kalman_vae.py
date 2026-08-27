"""Abstract base classes for the Kalman VAE components.

These define the interface that sport-specific implementations must follow.
The architecture is always:
    Encoder: transition_probs -> y_hat
    Kalman:  z_prev + y_hat    -> z_new
    Decoder: z                 -> reconstructed transition_probs
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


try:
    import torch.nn as nn
    _BaseModule = nn.Module
except ImportError:
    _BaseModule = object


class Encoder(ABC, _BaseModule):
    """Encode transition probabilities into an observation vector y_hat."""

    @abstractmethod
    def forward(self, transition_probs):
        """Input: [batch, input_dim] or [input_dim]. Output: y_hat."""
        pass


class KalmanFilter(ABC, _BaseModule):
    """Update team latent z_t using encoded observation y_hat."""

    @abstractmethod
    def predict(self, z_prev):
        """Predict step: z_t|t-1 from z_prev."""
        pass

    @abstractmethod
    def update(self, z_pred, y_hat):
        """Update step: incorporate observation y_hat."""
        pass

    @abstractmethod
    def forward(self, z_prev, y_hat):
        """Predict then update."""
        pass


class Decoder(ABC, _BaseModule):
    """Decode team latent into transition probabilities."""

    @abstractmethod
    def forward(self, z):
        """Input: [batch, z_dim] or [z_dim]. Output: logits (not softmaxed)."""
        pass


class MatchupPredictor(ABC, _BaseModule):
    """Mix two team latents and decode into predicted matchup transitions."""

    @abstractmethod
    def forward(self, z_home, z_away):
        """Input: two team latents. Output: predicted transition probabilities."""
        pass


class Trainer(ABC):
    """Train encoder + kalman + decoder on team game histories."""

    @abstractmethod
    def fit(self, team_games, n_epochs=10):
        """Train on {team_id: [game_transition_tensors]}. Returns epoch losses."""
        pass

    @abstractmethod
    def get_team_latent(self, team_id):
        """Get current latent for a team."""
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
