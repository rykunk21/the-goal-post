"""Sport-agnostic transition model abstract base class."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np


class TransitionModel(ABC):
    """Abstract base for sport-specific transition models.

    Each sport implements its own state space and transition extraction,
    but the interface is common:
    - Extract transitions from a single game for a single team
    - Get probability distributions over next states
    - Simulate drives/games using the extracted matrix
    - Evaluate reconstruction against actual results
    """

    def __init__(self, team_id: str, game_id: str, opponent_id: str):
        self.team_id = team_id
        self.game_id = game_id
        self.opponent_id = opponent_id

    @abstractmethod
    def extract_from_game(self, game) -> None:
        """Extract transition matrix from a specific game for this team.

        Each sport implements its own state discretization here.
        """
        pass

    @abstractmethod
    def get_state_transition_probabilities(self, state_key: str) -> Dict[str, float]:
        """Get P(next_state | current_state_key).

        Returns: {next_state_key: probability}
        State keys are sport-specific.
        """
        pass

    @abstractmethod
    def simulate_drive(self, start_state, max_plays: int = 20) -> Tuple[List[str], int, Optional[str]]:
        """Simulate a single possession/drive using this team's transition matrix.

        Returns: (state_sequence, points_scored, terminal_outcome)
        """
        pass

    @abstractmethod
    def simulate_game(self, n_possessions: int = 12) -> int:
        """Simulate this team's possessions using their own transition matrix.

        Returns: total points scored
        """
        pass

    @abstractmethod
    def evaluate_reconstruction(self, actual_points: int, n_sims: int = 1000) -> Dict[str, float]:
        """Evaluate how well this transition matrix reconstructs the actual game score.

        Simulate many times and check if actual score falls in distribution.
        """
        pass

    @abstractmethod
    def get_matrix_summary(self) -> Dict[str, any]:
        """Return summary statistics of this transition matrix.

        Used for dataset features/labels.
        """
        pass
