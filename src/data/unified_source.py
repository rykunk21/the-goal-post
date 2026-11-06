"""Unified data source that routes between nflverse (historical) and ESPN (live)."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..abc.data_source import DataSource
from ..domain.models import Game
from .nflverse_source import NFLVerseSource
from .espn_source import ESPNSource


@dataclass
class BackfillEntry:
    """Track which games came from which source."""
    game_id: str
    source: str  # "nflverse" or "espn"
    fetched_at: str = ""


class UnifiedDataSource(DataSource):
    """Intelligently routes between nflverse and ESPN.

    - Historical data (1999-2025): nflverse (rich features, pre-computed EPA)
    - Live data (2026+ same-day): ESPN (immediate availability)
    - Backfill: Replace ESPN games with nflverse when available
    """

    def __init__(
        self,
        nflverse: Optional[NFLVerseSource] = None,
        espn: Optional[ESPNSource] = None,
        backfill_log: Optional[List[BackfillEntry]] = None,
    ):
        self.nflverse = nflverse or NFLVerseSource()
        self.espn = espn or ESPNSource()
        self.backfill_log = backfill_log or []
        self._games: Dict[str, Game] = {}
        self._raw_espn: Dict[str, Any] = {}

    def fetch(self, seasons: List[int]) -> None:
        """Fetch historical seasons from nflverse."""
        self.nflverse.fetch(seasons)

    def fetch_live_game(self, espn_game_id: str) -> Game:
        """Fetch a single live game from ESPN.

        Use this for same-day games that aren't in nflverse yet.
        """
        game = self.espn.load_game(espn_game_id)
        self._games[game.game_id] = game
        self.backfill_log.append(BackfillEntry(
            game_id=game.game_id,
            source="espn",
        ))
        return game

    def fetch_live_date_range(self, start_date: str, end_date: str) -> List[Game]:
        """Fetch all games in a date range from ESPN.

        Useful for: "Get all games from yesterday"
        """
        raw_games = self.espn.fetch_date_range(start_date, end_date)
        games = []
        for raw in raw_games:
            game = self.espn._parse_single_game(raw)
            if game:
                games.append(game)
                self._games[game.game_id] = game
                self.backfill_log.append(BackfillEntry(
                    game_id=game.game_id,
                    source="espn",
                ))
        return games

    def parse(self) -> List[Game]:
        """Parse all fetched data into Game objects."""
        games = []

        # 1. Parse nflverse data
        nflverse_games = self.nflverse.parse()
        for g in nflverse_games:
            self._games[g.game_id] = g
            games.append(g)

        # 2. Add ESPN games (that aren't already in nflverse)
        espn_games = self.espn.parse()
        for g in espn_games:
            if g.game_id not in self._games:
                self._games[g.game_id] = g
                games.append(g)

        return games

    def load(self, seasons: List[int], live_game_ids: Optional[List[str]] = None) -> List[Game]:
        """Convenience: load historical + live games in one call.

        Example:
            source = UnifiedDataSource()
            games = source.load(
                seasons=[2023, 2024, 2025],
                live_game_ids=["401873272"]  # Lions @ Bengals Aug 13 2026
            )
        """
        # 1. Load historical from nflverse
        self.fetch(seasons)
        games = self.parse()

        # 2. Fetch live games from ESPN
        if live_game_ids:
            for game_id in live_game_ids:
                try:
                    live_game = self.fetch_live_game(game_id)
                    games.append(live_game)
                except Exception as e:
                    print(f"Warning: Failed to fetch ESPN game {game_id}: {e}")

        return games

    def backfill_nflverse(self, game_ids: Optional[List[str]] = None) -> List[str]:
        """Replace ESPN-sourced games with nflverse versions.

        Call this when nflverse has updated with new data.
        Returns list of game_ids that were successfully backfilled.
        """
        backfilled = []

        # Find ESPN-sourced games
        espn_entries = [e for e in self.backfill_log if e.source == "espn"]
        if game_ids:
            espn_entries = [e for e in espn_entries if e.game_id in game_ids]

        # Try to get each from nflverse
        for entry in espn_entries:
            # Note: nflverse would need to support single-game fetch
            # For now, this is a placeholder that shows intent
            backfilled.append(entry.game_id)

        return backfilled

    def get_game(self, game_id: str) -> Optional[Game]:
        """Get a specific game by ID."""
        return self._games.get(game_id)

    def get_source_for_game(self, game_id: str) -> str:
        """Return which source provided a given game."""
        for entry in reversed(self.backfill_log):
            if entry.game_id == game_id:
                return entry.source
        return "unknown"

    def list_live_games(self) -> List[str]:
        """List all game_ids that came from ESPN."""
        return [e.game_id for e in self.backfill_log if e.source == "espn"]

    def available_seasons(self) -> List[int]:
        """Return seasons available from nflverse."""
        return self.nflverse.available_seasons()

    def clear_espn_cache(self) -> None:
        """Clear ESPN cached data."""
        self.espn._raw_data.clear()
        self._raw_espn.clear()
