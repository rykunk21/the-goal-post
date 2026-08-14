"""NFLVerse data source implementation."""

from typing import List, Optional
import os

from ..abc.data_source import DataSource
from ..domain.models import Game, Possession, Play, PossessionResult


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
        """Convert nflverse pbp into Game/Possession/Play domain objects."""
        if self._raw_data is None:
            raise RuntimeError("No data fetched. Call fetch() first.")

        import polars as pl

        games = []
        for df in self._raw_data:
            # Filter out rows with null game_id and drive
            df = df.filter(
                pl.col("game_id").is_not_null() &
                pl.col("drive").is_not_null()
            )

            # Group by game_id
            game_ids = df["game_id"].unique().to_list()

            for game_id in game_ids:
                game_df = df.filter(pl.col("game_id") == game_id)

                # Get game-level metadata from first row
                first_row = game_df.row(0, named=True)
                season = int(first_row.get("season", 0))
                week = int(first_row.get("week", 0))
                home_team = str(first_row.get("home_team", ""))
                away_team = str(first_row.get("away_team", ""))

                # Build possessions from drives
                possessions = self._build_possessions(game_df)

                game = Game(
                    game_id=str(game_id),
                    season=season,
                    week=week,
                    home_team=home_team,
                    away_team=away_team,
                    possessions=possessions,
                    sport="nfl",
                )
                games.append(game)

        return games

    def _build_possessions(self, game_df) -> List[Possession]:
        """Group plays by drive within a game."""
        import polars as pl

        possessions = []

        # Sort by play_id to ensure chronological order
        game_df = game_df.sort("play_id")

        # Group by drive - filter out null drives
        game_df = game_df.filter(pl.col("drive").is_not_null())
        drive_groups = game_df.group_by(["drive"])

        for drive_num, drive_df in drive_groups:
            drive_val = drive_num[0] if isinstance(drive_num, tuple) else drive_num
            if drive_val is None:
                continue
            drive_num = int(drive_val)
            drive_id = f"{drive_df['game_id'][0]}_d{drive_num}"

            # Get team with possession from first play
            first_play = drive_df.row(0, named=True)
            posteam = str(first_play.get("posteam", "")) if first_play.get("posteam") else ""

            plays = []
            for row in drive_df.iter_rows(named=True):
                play = Play(
                    play_id=str(row.get("play_id", "")),
                    down=int(row.get("down", 0)) if row.get("down") is not None else None,
                    distance=int(row.get("ydstogo", 0)) if row.get("ydstogo") is not None else None,
                    yardline=int(row.get("yardline_100", 0)) if row.get("yardline_100") is not None else None,
                    play_type=str(row.get("play_type", "")) if row.get("play_type") else "",
                    yards_gained=int(row.get("yards_gained", 0)) if row.get("yards_gained") is not None else 0,
                    points_scored=6 if row.get("td_team") else 0,
                    epa=float(row.get("epa", 0.0)) if row.get("epa") is not None else 0.0,
                    wp=float(row.get("wp", 0.5)) if row.get("wp") is not None else 0.5,
                    passer=str(row.get("passer", "")) if row.get("passer") else None,
                    rusher=str(row.get("rusher", "")) if row.get("rusher") else None,
                    receiver=str(row.get("receiver", "")) if row.get("receiver") else None,
                    penalty=bool(row.get("penalty", False)) if row.get("penalty") is not None else False,
                    turnover=bool(row.get("turnover", False)) if row.get("turnover") is not None else False,
                    scoring_play=bool(row.get("td_team", False)) if row.get("td_team") is not None else False,
                )
                plays.append(play)

            # Determine drive result from last play
            result = self._infer_drive_result(plays)

            possession = Possession(
                possession_id=drive_id,
                team=posteam,
                plays=plays,
                result=result,
                quarter=int(first_play.get("qtr", 1)) if first_play.get("qtr") else 1,
                start_field_position=int(first_play.get("yardline_100", 25)) if first_play.get("yardline_100") else None,
                game_id=str(first_play.get("game_id", "")),
                sport="nfl",
            )
            possessions.append(possession)

        return possessions

    def _infer_drive_result(self, plays: List[Play]) -> Optional[PossessionResult]:
        """Infer the drive result from the last play."""
        if not plays:
            return None

        last_play = plays[-1]

        # Check for touchdown
        if last_play.scoring_play and last_play.points_scored >= 6:
            return PossessionResult.TOUCHDOWN

        # Check for field goal
        if last_play.scoring_play and last_play.points_scored == 3:
            return PossessionResult.FIELD_GOAL

        # Check for safety
        if last_play.scoring_play and last_play.points_scored == 2:
            return PossessionResult.SAFETY

        # Check for turnover
        if last_play.turnover:
            return PossessionResult.TURNOVER

        # Check for punt
        if last_play.play_type == "punt":
            return PossessionResult.PUNT

        # Check for turnover on downs
        if last_play.down == 4 and last_play.yards_gained is not None and last_play.distance is not None:
            if last_play.yards_gained < last_play.distance:
                return PossessionResult.TURNOVER_ON_DOWNS

        return None

    def available_seasons(self) -> List[int]:
        """Check cache for already-downloaded seasons."""
        return self._seasons
