"""TransitionModel abstract base class."""

from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

from ..domain.models import GameState, Transition, TransitionMatrix


class TransitionModel(ABC):
    """Predict state-to-state transition probabilities."""

    @abstractmethod
    def fit(
        self,
        representations: Dict[str, np.ndarray],
        transitions: List[Transition],
    ) -> "TransitionModel":
        """Train on team embeddings + observed transitions."""
        pass

    @abstractmethod
    def predict(
        self,
        z_home: np.ndarray,
        z_away: np.ndarray,
        state: GameState,
    ) -> TransitionMatrix:
        """Output transition probabilities for a given game state."""
        pass
