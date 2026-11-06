"""ESPN API data source for live/recent NFL games."""

from typing import List, Optional, Dict, Any
import os

from ..abc.data_source import DataSource
from ..domain.models import Game, Possession, Play, PossessionResult


class ESPNSource(DataSource):
    """Pull play-by-play data from ESPN's public API.

    Provides same-day game data when nflverse hasn't updated yet.
    Less rich than nflverse (no pre-computed EPA/WPA) but available immediately.
    """

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.goalpost/espn_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._raw_data: Dict[str, Any] = {}

    def fetch(self, seasons: List[int]) -> None:
        """Not implemented for ESPN — use fetch_game() for specific games."""
        raise NotImplementedError(
            "ESPNSource.fetch(seasons) not supported. "
            "Use fetch_game(game_id) or fetch_date_range(start, end)."
        )

    def fetch_game(self, game_id: str) -> Dict[str, Any]:
        """Fetch a specific game by ESPN event ID."""
        import requests

        url = f"{self.BASE_URL}/summary"
        params = {"event": game_id}

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        self._raw_data[game_id] = data
        return data

    def fetch_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch games between dates (YYYY-MM-DD format).

        Returns list of game data dicts that can be parsed individually.
        """
        import requests

        url = f"{self.BASE_URL}/scoreboard"
        params = {
            "dates": start_date.replace("-", ""),
            "limit": 100,
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        scoreboard = response.json()

        games = []
        for event in scoreboard.get("events", []):
            game_id = event.get("id")
            if game_id:
                game_data = self.fetch_game(game_id)
                games.append(game_data)

        return games

    def parse(self) -> List[Game]:
        """Parse all cached ESPN raw data into Game objects."""
        games = []
        for game_id, raw_data in self._raw_data.items():
            game = self._parse_single_game(raw_data)
            if game:
                games.append(game)
        return games

    def _parse_single_game(self, raw_data: Dict[str, Any]) -> Optional[Game]:
        """Convert ESPN game JSON into a Game domain object."""
        header = raw_data.get("header", {})
        competitions = header.get("competitions", [])
        if not competitions:
            return None

        comp = competitions[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            return None

        # Extract teams
        home_team = ""
        away_team = ""
        home_score = 0
        away_score = 0

        for c in competitors:
            team_data = c.get("team", {})
            abbrev = team_data.get("abbreviation", "")
            score = int(c.get("score", 0) or 0)

            if c.get("homeAway") == "home":
                home_team = abbrev
                home_score = score
            else:
                away_team = abbrev
                away_score = score

        # Game metadata
        game_id = str(raw_data.get("id", ""))
        season = header.get("season", {}).get("year", 0)
        week_raw = header.get("week", 0)
        week = week_raw.get("number", 0) if isinstance(week_raw, dict) else int(week_raw)

        # Extract possessions from drives
        possessions = self._extract_possessions(raw_data, home_team, away_team)

        return Game(
            game_id=game_id,
            season=int(season),
            week=int(week),
            home_team=home_team,
            away_team=away_team,
            possessions=possessions,
            home_score=home_score,
            away_score=away_score,
            sport="nfl",
        )

    def _extract_possessions(
        self, raw_data: Dict[str, Any], home_team: str, away_team: str
    ) -> List[Possession]:
        """Extract possessions (drives) from ESPN drive data."""
        drives_data = raw_data.get("drives", {})
        if isinstance(drives_data, dict) and "previous" in drives_data:
            drive_list = drives_data["previous"]
        elif isinstance(drives_data, list):
            drive_list = drives_data
        else:
            return []

        possessions = []
        for i, drive in enumerate(drive_list):
            team_data = drive.get("team", {})
            posteam = team_data.get("abbreviation", "")

            # Build drive ID
            game_id = str(raw_data.get("id", ""))
            drive_id = f"{game_id}_d{i+1}"

            # Extract plays
            plays = self._extract_plays(drive.get("plays", []))

            # Determine drive result
            result = self._infer_drive_result(drive)

            # Calculate points scored
            points_scored = sum(p.points_scored for p in plays)

            # Get field positions from first/last plays
            start_pos = None
            end_pos = None
            if plays:
                start_pos = plays[0].yardline
                end_pos = plays[-1].yardline

            # Determine quarter from first play
            quarter = 1
            if plays and hasattr(plays[0], "quarter"):
                quarter = plays[0].quarter or 1

            possession = Possession(
                possession_id=drive_id,
                team=posteam,
                plays=plays,
                result=result,
                quarter=quarter,
                start_field_position=start_pos,
                end_field_position=end_pos,
                points_scored=points_scored,
                game_id=game_id,
                sport="nfl",
            )
            possessions.append(possession)

        return possessions

    def _extract_plays(self, plays_data: List[Dict[str, Any]]) -> List[Play]:
        """Convert ESPN play data into Play domain objects."""
        plays = []

        for i, play in enumerate(plays_data):
            play_id = str(play.get("id", i))
            text = play.get("text", "")

            # Parse play type from text and structured data
            play_type = self._infer_play_type(play, text)

            # Extract yardline
            yardline = None
            start_info = play.get("start", {})
            if start_info:
                yardline_raw = start_info.get("yardLine")
                if yardline_raw is not None:
                    # ESPN yardline: 50 is midfield, <50 is own side
                    yardline = int(yardline_raw)

            # Extract down and distance
            down = play.get("down", None)
            distance = None
            if down:
                distance = play.get("distance", None)
                if distance:
                    distance = int(distance)
                down = int(down)

            # Extract yards gained
            yards_gained = play.get("statYardage", 0)
            if yards_gained is None:
                yards_gained = 0
            yards_gained = int(yards_gained)

            # Check for scoring
            scoring_play = bool(play.get("scoringPlay", False))
            points_scored = 0
            if scoring_play:
                points_scored = self._extract_points_from_play(play, text)

            # Check for turnover
            turnover = self._is_turnover(text, play)

            # Extract quarter/period
            quarter = play.get("period", {}).get("number", 1)
            if isinstance(quarter, dict):
                quarter = quarter.get("number", 1)
            quarter = int(quarter) if quarter else 1

            play_obj = Play(
                play_id=play_id,
                down=down,
                distance=distance,
                yardline=yardline,
                play_type=play_type,
                yards_gained=yards_gained,
                points_scored=points_scored,
                epa=0.0,  # Not available from ESPN
                wp=0.5,  # Not available from ESPN
                turnover=turnover,
                scoring_play=scoring_play,
            )
            plays.append(play_obj)

        return plays

    def _infer_play_type(self, play: Dict[str, Any], text: str) -> str:
        """Infer play type from ESPN play data and description text."""
        text_lower = text.lower()

        # Check ESPN type first
        play_type = play.get("type", {}).get("text", "")
        if play_type:
            return play_type.lower().replace(" ", "_")

        # Parse from text
        if "kickoff" in text_lower:
            return "kickoff"
        elif "punt" in text_lower:
            return "punt"
        elif "field goal" in text_lower:
            return "field_goal"
        elif "pass" in text_lower:
            return "pass"
        elif any(word in text_lower for word in ["run", "rush", "up the middle", "left guard", "right tackle"]):
            return "run"
        elif "sack" in text_lower:
            return "sack"
        elif "extra point" in text_lower:
            return "extra_point"
        elif "two-point" in text_lower or "2-point" in text_lower:
            return "two_point_conversion"
        elif "kneel" in text_lower:
            return "kneel"
        elif "spike" in text_lower:
            return "spike"
        elif "timeout" in text_lower:
            return "timeout"
        else:
            return "no_play"

    def _extract_points_from_play(self, play: Dict[str, Any], text: str) -> int:
        """Extract points scored on a play."""
        text_lower = text.lower()

        # Touchdown = 6 (before extra point)
        if "touchdown" in text_lower or "td" in text_lower:
            return 6
        elif "field goal" in text_lower and "good" in text_lower:
            return 3
        elif "safety" in text_lower:
            return 2
        elif "extra point" in text_lower and "good" in text_lower:
            return 1
        elif ("two-point" in text_lower or "2-point" in text_lower) and "good" in text_lower:
            return 2

        # Fallback: check ESPN scoring type
        scoring_type = play.get("scoringType", {}).get("name", "").lower()
        if scoring_type == "touchdown":
            return 6
        elif scoring_type == "fieldgoal":
            return 3
        elif scoring_type == "safety":
            return 2
        elif scoring_type == "extrapoint":
            return 1

        return 0

    def _is_turnover(self, text: str, play: Dict[str, Any]) -> bool:
        """Check if a play resulted in a turnover."""
        text_lower = text.lower()

        # Check explicit turnover flag if available
        if play.get("turnover", False):
            return True

        # Check text for turnover indicators
        turnover_indicators = [
            "interception", "intercepted", "fumble", "fumbled",
            "recovered by", "muffed", "turnover on downs"
        ]
        return any(indicator in text_lower for indicator in turnover_indicators)

    def _infer_drive_result(self, drive: Dict[str, Any]) -> Optional[PossessionResult]:
        """Map ESPN drive result to PossessionResult enum."""
        result = drive.get("displayResult", "")
        if not result:
            result = drive.get("result", "")

        result_str = str(result).lower().strip()

        mapping = {
            "touchdown": PossessionResult.TOUCHDOWN,
            "field goal": PossessionResult.FIELD_GOAL,
            "punt": PossessionResult.PUNT,
            "interception": PossessionResult.TURNOVER,
            "fumble": PossessionResult.TURNOVER,
            "turnover": PossessionResult.TURNOVER,
            "downs": PossessionResult.TURNOVER_ON_DOWNS,
            "turnover on downs": PossessionResult.TURNOVER_ON_DOWNS,
            "safety": PossessionResult.SAFETY,
            "end of half": PossessionResult.END_OF_HALF,
            "end of game": PossessionResult.END_OF_GAME,
            "missed field goal": PossessionResult.TURNOVER,
        }

        return mapping.get(result_str)

    def available_seasons(self) -> List[int]:
        """ESPN doesn't cache seasons in the same way — return empty."""
        return []

    def load_game(self, game_id: str) -> Game:
        """Convenience: fetch + parse a single game."""
        self.fetch_game(game_id)
        games = self.parse()
        if games:
            return games[0]
        raise RuntimeError(f"Failed to parse game {game_id}")
