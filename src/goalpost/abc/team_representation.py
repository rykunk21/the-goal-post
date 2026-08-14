"""TeamRepresentation abstract base class."""

from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

from ..domain.models import Game, Possession, GameContext


class TeamRepresentation(ABC):
    """Learn team embeddings from historical performance."""

    @abstractmethod
    def fit(self, possessions: List[Possession]) -> "TeamRepresentation":
        """Learn representations from historical possessions."""
        pass

    @abstractmethod
    def encode(self, team_id: str, context: GameContext) -> np.ndarray:
        """Emit a latent vector for a team given game context."""
        pass

    @abstractmethod
    def update(self, game: Game) -> "TeamRepresentation":
        """In-season update (e.g. Bayesian) from a newly completed game."""
        pass
