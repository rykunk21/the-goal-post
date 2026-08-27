"""Abstract base class for data sources."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..domain.models import Game


class DataSource(ABC):
    """Pull raw game data from an external provider.

    Sport-specific implementations handle provider quirks (nflverse,
    ESPN, StatsBroadcast XML, etc.) but all emit the same domain model.
    """

    @abstractmethod
    def fetch(self, **kwargs) -> None:
        """Fetch raw records from the provider. Store internally."""
        pass

    @abstractmethod
    def parse(self) -> List[Game]:
        """Convert fetched raw records into domain Game objects."""
        pass

    def load(self, **kwargs) -> List[Game]:
        """Convenience: fetch + parse."""
        self.fetch(**kwargs)
        return self.parse()
