# Purpose: select completed and upcoming rows from a caller-supplied NFL schedule CSV.
# Public API: NFLSchedule.completed_games and NFLSchedule.upcoming return full row mappings.
# Consumers: NFLDataSource uses upcoming; completed_games is available to source clients
# but has no current production call site in this repository.

from datetime import date, datetime
from pathlib import Path
from typing import Any, List

import polars as pl


# Purpose: retain a loaded schedule for repeated local row selection without fetching data.
# Consumers: NFLDataSource owns an instance; source clients can use it directly.
class NFLSchedule:
    # Purpose: load the supplied CSV using Polars' existing column inference.
    # Callers: NFLDataSource construction and direct schedule clients.
    # This boundary does not normalize dates, timezones or provider-specific columns.
    def __init__(self, path: Path) -> None:
        self.df: pl.DataFrame = pl.read_csv(path)

    # Purpose: select a season's non-preseason rows with both recorded scores present.
    # Callers: direct schedule clients; no current repository call site invokes this method.
    # Score presence is the selection rule, not independent verification of a final result.
    def completed_games(self, season: int) -> List[dict[str, Any]]:
        d = self.df.filter(
            (pl.col("season") == season)
            & (pl.col("home_score").is_not_null())
            & (pl.col("away_score").is_not_null())
            & (pl.col("game_type") != "PRE")
        )
        return d.to_dicts()

    # Purpose: select rows strictly after as_of whose home score is absent.
    # Callers: NFLDataSource.fetch_upcoming and direct schedule clients.
    # Comparison uses the stored gameday type directly; callers must supply a compatible
    # cutoff. No date coercion, timezone conversion or separate away-score check is added.
    def upcoming(self, as_of: date | datetime | str) -> List[dict[str, Any]]:
        d = self.df.filter(
            (pl.col("gameday") > as_of)
            & (pl.col("home_score").is_null())
        )
        return d.to_dicts()
