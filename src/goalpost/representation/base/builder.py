# Purpose: Define the sport-neutral boundary between extracted games and transition tables.
# Public API: GameTable marker and abstract TransitionBuilder.build.
# Consumers: NFLGameTable/NFLTransitionBuilder and shared builder-contract tests.

from abc import ABC, abstractmethod
from typing import List

from .models import ExtractedGame


# Purpose: Mark builder results while leaving storage and indexing to each sport.
# Consumers: NFLGameTable subclasses; TransitionBuilder return annotations.
class GameTable:
    pass


# Purpose: Require sport builders to turn extracted games into a reusable table.
# Consumers: NFLTransitionBuilder, CLI dataset construction and abstract-contract tests.
class TransitionBuilder(ABC):
    # Purpose: Build a sport-specific table from extracted records without prescribing layout.
    # Callers: CLI dataset construction through concrete builder implementations.
    @abstractmethod
    def build(self, games: List[ExtractedGame]) -> GameTable:
        pass
