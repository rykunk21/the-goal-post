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
            # Use fixed_drive (cleaned) instead of drive (raw)
            # Filter out rows with null game_id and fixed_drive
            df = df.filter(
                pl.col("game_id").is_not_null() &
                pl.col("fixed_drive").is_not_null()
            )

            # Get game-level metadata
            game_ids = df["game_id"].unique().to_list()

            for game_id in game_ids:
                game_df = df.filter(pl.col("game_id") == game_id)

                # Get game-level metadata from first row
                first_row = game_df.row(0, named=True)
                season = int(first_row.get("season", 0))
                week = int(first_row.get("week", 0))
                home_team = str(first_row.get("home_team", ""))
                away_team = str(first_row.get("away_team", ""))
                home_score = int(first_row.get("home_score", 0) or 0)
                away_score = int(first_row.get("away_score", 0) or 0)

                # Build possessions from drives
                possessions = self._build_possessions(game_df)

                game = Game(
                    game_id=str(game_id),
                    season=season,
                    week=week,
                    home_team=home_team,
                    away_team=away_team,
                    possessions=possessions,
                    home_score=home_score,
                    away_score=away_score,
                    sport="nfl",
                )
                games.append(game)

        return games

    def _build_possessions(self, game_df) -> List[Possession]:
        """Group plays by fixed_drive within a game."""
        import polars as pl

        possessions = []

        # Sort by play_id BEFORE grouping to ensure chronological order
        game_df = game_df.sort("play_id")

        # Group by fixed_drive (cleaned drive number)
        game_df = game_df.filter(pl.col("fixed_drive").is_not_null())

        # Use group_by with maintain_order=True to preserve sort order
        for drive_num, drive_df in game_df.group_by("fixed_drive", maintain_order=True):
            drive_val = drive_num[0] if isinstance(drive_num, tuple) else drive_num
            if drive_val is None:
                continue
            drive_num = int(drive_val)
            drive_id = f"{game_df['game_id'][0]}_d{drive_num}"

            # Get team with possession from first play with valid posteam
            posteam = ""
            first_play_row = None
            for row in drive_df.iter_rows(named=True):
                if row.get("posteam"):
                    posteam = str(row.get("posteam"))
                    first_play_row = row
                    break

            if first_play_row is None:
                continue  # Skip drives with no valid posteam

            plays = []
            for row in drive_df.iter_rows(named=True):
                play = Play(
                    play_id=str(row.get("play_id", "")),
                    down=int(row.get("down", 0)) if row.get("down") is not None else None,
                    distance=int(row.get("ydstogo", 0)) if row.get("ydstogo") is not None else None,
                    yardline=int(row.get("yardline_100", 0)) if row.get("yardline_100") is not None else None,
                    play_type=str(row.get("play_type", "")) if row.get("play_type") else "",
                    yards_gained=int(row.get("yards_gained", 0)) if row.get("yards_gained") is not None else 0,
                    points_scored=self._extract_points(row),
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

            # Determine drive result from nflverse's fixed_drive_result
            result = self._infer_drive_result(first_play_row, plays)

            # Calculate points scored on this drive
            points_scored = sum(p.points_scored for p in plays)

            # Get end field position from last play
            end_field_position = plays[-1].yardline if plays else None

            possession = Possession(
                possession_id=drive_id,
                team=posteam,
                plays=plays,
                result=result,
                quarter=int(first_play_row.get("qtr", 1)) if first_play_row.get("qtr") else 1,
                start_field_position=int(first_play_row.get("yardline_100", 25)) if first_play_row.get("yardline_100") else None,
                end_field_position=end_field_position,
                points_scored=points_scored,
                game_id=str(first_play_row.get("game_id", "")),
                sport="nfl",
            )
            possessions.append(possession)

        return possessions

    def _extract_points(self, row: dict) -> int:
        """Extract points scored on a play."""
        # Touchdown = 6
        if row.get("touchdown") and row.get("td_team"):
            return 6
        # Field goal = 3
        if row.get("field_goal_result") == "made":
            return 3
        # Safety = 2
        if row.get("safety"):
            return 2
        # Extra point = 1 (not typically part of a drive's main sequence)
        if row.get("extra_point_result") == "good":
            return 1
        # 2-point conversion = 2
        if row.get("two_point_conv_result") == "success":
            return 2
        return 0

    def _infer_drive_result(
        self, first_play_row: Optional[dict], plays: List[Play]
    ) -> Optional[PossessionResult]:
        """Infer the drive result from nflverse's fixed_drive_result column."""
        if first_play_row is None:
            return None

        result_str = first_play_row.get("fixed_drive_result", "")
        if not result_str:
            return None

        result_str = str(result_str).lower().strip()

        # Map nflverse drive results to PossessionResult enum
        mapping = {
            "touchdown": PossessionResult.TOUCHDOWN,
            "field goal": PossessionResult.FIELD_GOAL,
            "missed field goal": PossessionResult.TURNOVER,  # Turnover on missed FG
            "punt": PossessionResult.PUNT,
            "turnover": PossessionResult.TURNOVER,
            "turnover on downs": PossessionResult.TURNOVER_ON_DOWNS,
            "safety": PossessionResult.SAFETY,
            "end of half": PossessionResult.END_OF_HALF,
            "end of game": PossessionResult.END_OF_GAME,
            "opp touchdown": PossessionResult.TURNOVER,  # Defensive/special teams TD
        }

        return mapping.get(result_str)

    def available_seasons(self) -> List[int]:
        """Check cache for already-downloaded seasons."""
        return self._seasons
