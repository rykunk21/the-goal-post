"""DataSource abstract base class."""

from abc import ABC, abstractmethod
from typing import List

from ..domain.models import Game


class DataSource(ABC):
    """Pull raw data from an external source and parse into domain objects."""

    @abstractmethod
    def fetch(self, seasons: List[int]) -> None:
        """Download/cache raw data for the given seasons."""
        pass

    @abstractmethod
    def parse(self) -> List[Game]:
        """Convert cached raw data into a list of Game domain objects."""
        pass

    @abstractmethod
    def available_seasons(self) -> List[int]:
        """Return seasons already cached/available locally."""
        pass

    def load(self, seasons: List[int]) -> List[Game]:
        """Convenience: fetch + parse in one call."""
        self.fetch(seasons)
        return self.parse()