# Purpose: Define the shared boundary from raw play data to extracted game records.
# Public API: GameExtractor.extract; ExtractedGame is imported for its return contract.
# Consumers: NFLGameExtractor, representation package exports and toy extractor tests.

from abc import ABC, abstractmethod
from typing import Any

import polars as pl

from .models import ExtractedGame


# Purpose: Require sport adapters to translate raw rows without prescribing event semantics.
# Consumers: NFLGameExtractor and test doubles implementing the extraction interface.
class GameExtractor(ABC):
    # Purpose: Convert raw rows and sport-specific metadata into one extracted game.
    # Callers: Dataset extraction integrations and extractor contract tests.
    @abstractmethod
    def extract(self, raw_df: pl.DataFrame, metadata: dict[str, Any]) -> ExtractedGame:
        pass
