"""NFLVerse data source implementation."""

from typing import List, Optional
import os

from ..abc.data_source import DataSource
from ..domain.models import Game, Drive, Play


class NFLVerseSource(DataSource):
    """Pull play-by-play data from nflverse/nflreadpy."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.goalpost/cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._raw_data = None
        self._seasons: List[int] = []

    def fetch(self, seasons: List[int]) -> None:
        """Download pbp data for given seasons via nflreadpy."""
        try:
            import nflreadpy
        except ImportError:
            raise ImportError("nflreadpy required. Install: pip install nflreadpy")

        dfs = []
        for season in seasons:
            df = nflreadpy.load_pbp(seasons=[season])
            dfs.append(df)
        self._raw_data = dfs
        self._seasons = seasons

    def parse(self) -> List[Game]:
        """Convert nflverse pbp into Game/Drive/Play domain objects."""
        # TODO: implement conversion from polars DataFrame to domain models
        raise NotImplementedError("parse() not yet implemented")

    def available_seasons(self) -> List[int]:
        """Check cache for already-downloaded seasons."""
        # TODO: check cache files
        return self._seasons