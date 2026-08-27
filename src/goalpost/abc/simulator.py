"""Abstract base class for simulators."""

from abc import ABC, abstractmethod
from typing import List

from ..domain.models import GameOutcome


class Simulator(ABC):
    """Run forward simulations from a transition model to produce outcomes."""

    @abstractmethod
    def simulate(self, transition_model, n_sims: int = 10000) -> List[GameOutcome]:
        """Run Monte Carlo forward to produce outcome distribution."""
        pass
