# Purpose: download public nflverse season files and check the required input columns.
# Public API: fetch_nflverse, download_season, NFLverseError, REQUIRED_COLUMNS, NFLVERSE_URL.
# Consumers: CLI dataset builds, the TUI download action and NFLDataSource.fetch.
# Validation checks column presence only; it does not certify results or every parser's needs.

from pathlib import Path
from typing import List

import polars as pl

NFLVERSE_URL: str = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/"
    "play_by_play_{season}.parquet"
)

REQUIRED_COLUMNS: set[str] = {
    "game_id",
    "home_team",
    "away_team",
    "posteam",
    "down",
    "ydstogo",
    "yardline_100",
    "half_seconds_remaining",
    "qtr",
    "play_type",
    "play_id",
    "posteam_score_post",
    "defteam_score_post",
    "home_score",
    "away_score",
    "week",
    "game_date",
}


# Purpose: identify download or required-column failures without substituting synthetic data.
# Consumers: fetcher callers and offline download/schema regression tests.
class NFLverseError(Exception):
    pass


# Purpose: reject a downloaded frame that lacks columns required by the extraction boundary.
# Callers: download_season before writing the season file; value/type checks are separate.
def _validate_nflverse_schema(df: pl.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise NFLverseError(
            f"Downloaded nflverse data is missing required columns: {sorted(missing)}"
        )


# Purpose: read one release parquet, validate its columns and write a compressed local copy.
# Callers: fetch_nflverse and offline fetcher tests; direct callers supply an existing dest.
# The same season path is replaced on a successful refresh; there is no cache-skip or retry.
def download_season(season: int, dest: Path) -> Path:
    url = NFLVERSE_URL.format(season=season)
    try:
        df = pl.read_parquet(url)
    except Exception as e:
        raise NFLverseError(
            f"Failed to download nflverse data for season {season} from {url}: {e}"
        ) from e
    _validate_nflverse_schema(df)
    out = dest / f"nfl-{season}.parquet"
    df.write_parquet(out, compression="zstd")
    return out


# Purpose: create the destination and download requested seasons in their supplied order.
# Callers: cli.build, tui.main's download action and NFLDataSource.fetch.
# The first failure stops the batch; files already written by earlier seasons remain.
def fetch_nflverse(seasons: List[int], dest: Path) -> List[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    return [download_season(s, dest) for s in seasons]
