# Purpose: Define sport-neutral extracted-game records and transition-model interfaces.
# Public API: ExtractedGame and the abstract TransitionModel contract.
# Consumers: Sport extractors/models, simulation engines and shared test doubles.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


# Purpose: Carry an identity and sport-specific events without imposing an event schema.
# Consumers: Game extractors, transition builders and toy fixtures.
@dataclass
class ExtractedGame:
    game_id: str = ""
    events: list[Any] = field(default_factory=list)


# Purpose: Separate transition storage/encoding from the simulator consuming probabilities.
# Consumers: NFLTransitionModel, simulation engines and ToyTransitionModel tests.
class TransitionModel(ABC):
    # Purpose: Bind the model to one team/game/opponent without allocating sport-specific data.
    # Callers: Concrete model constructors through super().__init__.
    def __init__(self, team_id: str, game_id: str, opponent_id: str) -> None:
        self.team_id = team_id
        self.game_id = game_id
        self.opponent_id = opponent_id

    # Purpose: Expose the number of state indices supported by the concrete model.
    # Callers: Simulation consumers and model-contract tests.
    @property
    @abstractmethod
    def state_space_size(self) -> int:
        pass

    # Purpose: Expose the number of conditioning contexts in the sport representation.
    # Callers: Simulation consumers and model-contract tests.
    @property
    @abstractmethod
    def context_count(self) -> int:
        pass

    # Purpose: Identify structurally allowed transitions independently of observed probabilities.
    # Callers: Model reconstruction and structural model checks.
    @property
    @abstractmethod
    def validity_mask(self) -> np.ndarray:
        pass

    # Purpose: Reserve the sport-specific extraction hook without defining event semantics.
    # Callers: Extraction integrations and test doubles; implementations supply the behavior.
    @abstractmethod
    def extract_from_game(self, game: ExtractedGame) -> None:
        pass

    # Purpose: Return destination probabilities for one context/state pair.
    # Callers: Simulation engines sampling the next transition.
    @abstractmethod
    def get_probabilities(self, context: int, state: int) -> Dict[int, float]:
        pass

    # Purpose: Encode model data as the concrete sport format used for persistence.
    # Callers: Dataset/serialization integrations and model round-trip tests.
    @abstractmethod
    def flatten(self) -> np.ndarray:
        pass

    # Purpose: Reconstruct a model using its sport-specific flat layout and supplied identity.
    # Callers: CLI simulation loading and model round-trip tests.
    @classmethod
    @abstractmethod
    def from_flat(
        cls, array: np.ndarray, team_id: str, game_id: str, opponent_id: str
    ) -> "TransitionModel":
        pass
