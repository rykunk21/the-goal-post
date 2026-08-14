"""DriveExtractor abstract base class."""

from abc import ABC, abstractmethod
from typing import List

from ..domain.models import Game, Drive, Transition


class DriveExtractor(ABC):
    """Group plays into drives and compute state transitions."""

    @abstractmethod
    def extract(self, games: List[Game]) -> List[Drive]:
        """Group plays into drives."""
        pass

    @abstractmethod
    def compute_state_transitions(self, drives: List[Drive]) -> List[Transition]:
        """Convert drives into (state, action, next_state) tuples."""
        pass

    def process(self, games: List[Game]) -> List[Transition]:
        """Convenience: extract drives + compute transitions."""
        drives = self.extract(games)
        return self.compute_state_transitions(drives)
