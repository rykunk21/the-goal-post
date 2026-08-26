"""TransitionModel abstract base classes."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..domain.models import GameState, Transition, TransitionMatrix


class PlayTransitionModel(ABC):
    """Predict play-to-play state transitions.

    Given current game state (down, distance, yardline) and play call,
    predict the distribution over next states.
    """

    @abstractmethod
    def fit(
        self,
        representations: Dict[str, np.ndarray],
        transitions: List[Transition],
    ) -> "PlayTransitionModel":
        """Train on team embeddings + observed play-level transitions."""
        pass

    @abstractmethod
    def predict(
        self,
        z_offense: np.ndarray,
        z_defense: np.ndarray,
        state: GameState,
        play_call: str,
    ) -> Dict[Tuple[Optional[int], Optional[int], Optional[int]], float]:
        """Predict next (down, distance, yardline) distribution.

        Returns: {(down, distance, yardline): probability}
        """
        pass


class DriveTransitionModel(ABC):
    """Predict drive-to-drive result transitions.

    Given drive start state and team matchup,
    predict the distribution over drive end results.
    """

    @abstractmethod
    def fit(
        self,
        representations: Dict[str, np.ndarray],
        transitions: List[Transition],
    ) -> "DriveTransitionModel":
        """Train on team embeddings + observed drive-level transitions."""
        pass

    @abstractmethod
    def predict(
        self,
        z_home: np.ndarray,
        z_away: np.ndarray,
        state: GameState,
    ) -> TransitionMatrix:
        """Predict drive result probabilities.

        Returns: {result: probability} where result is a drive outcome string.
        """
        pass
