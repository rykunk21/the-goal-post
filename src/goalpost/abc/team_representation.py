"""TeamRepresentation abstract base class."""

from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

from ..domain.models import Game, Drive, GameContext


class TeamRepresentation(ABC):
    """Learn team embeddings from historical performance."""

    @abstractmethod
    def fit(self, drives: List[Drive]) -> "TeamRepresentation":
        """Learn representations from historical drives."""
        pass

    @abstractmethod
    def encode(self, team_id: str, context: GameContext) -> np.ndarray:
        """Emit a latent vector for a team given game context."""
        pass

    @abstractmethod
    def update(self, game: Game) -> "TeamRepresentation":
        """In-season update (e.g. Bayesian) from a newly completed game."""
        pass
