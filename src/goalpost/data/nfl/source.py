# Purpose: adapt a local schedule and cached season parquets to the shared game-data interface.
# Public API: NFLDataSource implements fetch, fetch_upcoming and parse, and inherits load.
# Consumers: data.nfl re-exports the adapter to source clients; current CLI builds use the
# fetcher directly rather than invoking this adapter's load or upcoming-game methods.

from datetime import datetime
from pathlib import Path
from typing import List

import polars as pl

from ..base.models import Game, Play, Possession
from ..base.source import DataSource
from .fetcher import fetch_nflverse
from .schedule import NFLSchedule


# Purpose: keep schedule lookup, season acquisition and cache parsing behind DataSource.
# Consumers: source clients through the data.nfl package export and inherited load method.
class NFLDataSource(DataSource):
    # Purpose: read the supplied schedule and retain the caller-selected cache location.
    # Callers: clients constructing an NFL source; no season download occurs here.
    def __init__(self, schedule_path: Path, cache_dir: Path) -> None:
        self.schedule: NFLSchedule = NFLSchedule(schedule_path)
        self.cache_dir: Path = cache_dir
        # Reserved legacy storage; current parsing reads files rather than this container.
        self._raw: list[object] = []

    # Purpose: populate or refresh the cache for the explicitly requested seasons.
    # Callers: inherited DataSource.load and clients explicitly requesting acquisition.
    def fetch(self, seasons: List[int]) -> None:
        fetch_nflverse(seasons, self.cache_dir)

    # Purpose: project schedule rows selected by NFLSchedule.upcoming into Game records.
    # Callers: source clients; as_of filtering and date representation follow NFLSchedule.
    # Scores/possessions retain Game defaults; these records are not completed outcomes.
    def fetch_upcoming(self, as_of: datetime) -> List[Game]:
        rows = self.schedule.upcoming(as_of)
        return [
            Game(
                game_id=str(r["game_id"]),
                season=int(r["season"]),
                week=int(r["week"]),
                home_team=str(r["home_team"]),
                away_team=str(r["away_team"]),
                date=datetime.fromisoformat(r["gameday"]),
            )
            for r in rows
        ]

    # Purpose: group every cached nfl-*.parquet by game and drive into shared game records.
    # Callers: inherited DataSource.load and clients parsing an existing cache directly.
    # Includes previously cached seasons. Input must also supply drive and season columns;
    # downloader column validation alone does not establish this parser's full input contract.
    # Game scores and possession-team labels use the first stored game row as implemented;
    # this adapter does not reconstruct final settlement results or infer missing play values.
    def parse(self) -> List[Game]:
        games = []
        for path in self.cache_dir.glob("nfl-*.parquet"):
            df = pl.read_parquet(path)
            for gid, g in df.group_by("game_id"):
                first = g.row(0, named=True)
                possessions = []
                for _, plays in g.group_by("drive"):
                    possession_plays = [
                        Play(play_id=str(p["play_id"]), play_type=str(p["play_type"]))
                        for p in plays.to_dicts()
                    ]
                    possessions.append(
                        Possession(
                            team=str(first["posteam"]),
                            plays=possession_plays,
                            game_id=str(gid),
                        )
                    )
                games.append(
                    Game(
                        game_id=str(gid),
                        season=int(first["season"]),
                        week=int(first["week"]),
                        home_team=str(first["home_team"]),
                        away_team=str(first["away_team"]),
                        possessions=possessions,
                        home_score=int(first["home_score"]),
                        away_score=int(first["away_score"]),
                    )
                )
        return games
