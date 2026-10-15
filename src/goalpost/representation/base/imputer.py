# Purpose: Define the abstract imputation boundary and minimal legacy container types.
# Public API: ImputedDataset, History and ImputationBuilder.build.
# Consumers: NFL imputation subclasses and base interface tests.

from abc import ABC, abstractmethod
from typing import Any

from .builder import GameTable


# Purpose: Provide the marker base for sport-specific imputed outputs.
# Consumers: NFLImputedDataset subclasses this otherwise empty interface.
class ImputedDataset:
    pass


# Purpose: Retain the minimal date/matrix pair container for the abstract interface.
# Consumers: ImputationBuilder annotations; NFL uses the richer base.history.History.
class History:
    # Purpose: Start a separate empty entry list for each history instance.
    # Callers: Consumers of this minimal history container; no direct repository callers.
    def __init__(self) -> None:
        self.entries: list[tuple[Any, Any]] = []

    # Purpose: Preserve supplied date/matrix values in insertion order without validation.
    # Callers: Consumers populating this minimal container; no direct repository callers.
    # Any preserves the unconstrained legacy boundary rather than asserting a tensor type.
    def add(self, date: Any, matrix: Any) -> None:
        self.entries.append((date, matrix))


# Purpose: Require sport adapters to produce imputed output from tables and history.
# Consumers: NFLImputationBuilder implements this abstract boundary.
class ImputationBuilder(ABC):
    # Purpose: Specify the build entrypoint while leaving fallback rules to subclasses.
    # Callers: Imputation pipelines invoking a concrete sport-specific builder.
    @abstractmethod
    def build(self, tables: GameTable, history: History) -> ImputedDataset:
        pass
