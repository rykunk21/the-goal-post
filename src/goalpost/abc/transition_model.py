"""Abstract base class for sport-specific transition models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from ..domain.models import Game


class TransitionModel(ABC):
    """Extract and simulate transitions from a single game for a single team.

    Each sport implements its own state space and transition extraction,
    but the interface is common:
    - Extract transitions from a game
    - Get probability distributions over next states
    - Simulate possessions using the extracted matrix
    - Evaluate reconstruction against actual results
    """

    def __init__(self, team_id: str, game_id: str, opponent_id: str):
        self.team_id = team_id
        self.game_id = game_id
        self.opponent_id = opponent_id

    @abstractmethod
    def extract_from_game(self, game: Game) -> None:
        """Extract transition matrix from a specific game for this team."""
        pass

    @abstractmethod
    def get_state_transition_probabilities(self, state_key: str) -> Dict[str, float]:
        """Get P(next_state | current_state_key)."""
        pass

    @abstractmethod
    def simulate_drive(self, start_state, max_plays: int = 20) -> Tuple[List[str], int, Optional[str]]:
        """Simulate a single possession using this team's transition matrix.

        Returns: (state_sequence, points_scored, terminal_outcome)
        """
        pass

    @abstractmethod
    def simulate_game(self, n_possessions: int = 12) -> int:
        """Simulate this team's possessions using their own transition matrix."""
        pass

    @abstractmethod
    def evaluate_reconstruction(self, actual_points: int, n_sims: int = 1000) -> Dict[str, float]:
        """Evaluate how well this transition matrix reconstructs the actual game score."""
        pass

    @abstractmethod
    def get_matrix_summary(self) -> Dict:
        """Return summary statistics of this transition matrix."""
        pass

    @abstractmethod
    def flatten_probabilities(self) -> List[float]:
        """Flatten transition probabilities into a vector for the encoder.

        Returns a fixed-size vector regardless of which states were observed.
        Unobserved states get probability 0.
        """
        pass

    @classmethod
    @abstractmethod
    def from_flat_probabilities(cls, probs: List[float], team_id: str, game_id: str, opponent_id: str):
        """Reconstruct a TransitionModel from a flattened probability vector.

        Used at inference time when the decoder outputs predicted probabilities.
        """
        pass
