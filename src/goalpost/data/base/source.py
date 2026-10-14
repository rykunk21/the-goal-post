# Purpose: separate provider acquisition, cached parsing and upcoming-game lookup.
# Public API: DataSource defines fetch, fetch_upcoming and parse; load composes fetch/parse.
# Consumers: NFLDataSource implements this interface; data.base re-exports it to clients.
# Concrete sources own storage and provider behavior; the base class performs no I/O itself.

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from .models import Game


# Purpose: give sport-specific adapters a common typed game-data interface.
# Consumers: NFLDataSource subclasses it; base-source tests verify it cannot be instantiated.
class DataSource(ABC):
    # Purpose: acquire the requested seasons in storage controlled by the concrete adapter.
    # Callers: load dispatches here before parsing; adapters may also expose direct refreshes.
    @abstractmethod
    def fetch(self, seasons: List[int]) -> None:
        pass

    # Purpose: expose schedule-based upcoming games separately from historical cache loading.
    # Callers: source clients can request this lookup; no current CLI call site uses it.
    # The concrete adapter defines how as_of is compared with its schedule timestamps.
    @abstractmethod
    def fetch_upcoming(self, as_of: datetime) -> List[Game]:
        pass

    # Purpose: turn the adapter's available stored observations into typed game records.
    # Callers: load dispatches here after fetch; concrete adapters determine the cache scope.
    @abstractmethod
    def parse(self) -> List[Game]:
        pass

    # Purpose: provide the shared acquisition-then-parse sequence for historical source clients.
    # Callers: clients may use this convenience method; no current CLI call site invokes it.
    # Parsing has no seasons parameter, so it may include other seasons already cached.
    def load(self, seasons: List[int]) -> List[Game]:
        self.fetch(seasons)
        return self.parse()
