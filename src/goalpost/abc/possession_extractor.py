"""Abstract base class for possession extractors."""

from abc import ABC, abstractmethod
from typing import List

from ..domain.models import Game, Possession, Transition


class PossessionExtractor(ABC):
    """Group plays into possessions and compute state transitions.

    A possession is a sport-agnostic term for a contiguous sequence of plays
    where one team controls the ball / puck / etc.:
    - NFL: "drive"
    - NBA: "possession"
    - MLB: "inning half"
    - NHL/Soccer: "shift" or "zone possession"
    """

    @abstractmethod
    def extract(self, games: List[Game]) -> List[Possession]:
        """Group plays into possessions."""
        pass

    @abstractmethod
    def compute_state_transitions(self, possessions: List[Possession]) -> List[Transition]:
        """Convert possessions into (state, action, next_state) tuples."""
        pass

    def process(self, games: List[Game]) -> List[Transition]:
        """Convenience: extract possessions + compute transitions."""
        possessions = self.extract(games)
        return self.compute_state_transitions(possessions)
