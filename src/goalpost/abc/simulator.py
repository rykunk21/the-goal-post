"""Simulator abstract base class."""

from abc import ABC, abstractmethod
from typing import List

from ..domain.models import GameState, GameOutcome, Market


class Simulator(ABC):
    """Run forward simulations from a transition model."""

    @abstractmethod
    def simulate(
        self,
        transition_model,
        initial_state: GameState,
        n_sims: int = 10000,
    ) -> List[GameOutcome]:
        """Run Monte Carlo forward to produce outcome distribution."""
        pass

    @abstractmethod
    def price(
        self,
        outcomes: List[GameOutcome],
        market: Market,
    ) -> tuple:
        """Price a market given simulated outcomes.

        Returns (probability, expected_value).
        """
        pass
