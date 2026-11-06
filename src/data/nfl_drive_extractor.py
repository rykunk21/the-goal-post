"""NFL drive extractor implementation."""

from typing import List, Tuple

from ..abc.possession_extractor import PossessionExtractor
from ..domain.models import Game, Possession, Play, Transition, GameState, PossessionResult


class NFLDriveExtractor(PossessionExtractor):
    """Group nflverse plays into drives (NFL-specific term for possessions)."""

    def extract(self, games: List[Game]) -> List[Possession]:
        """Extract possessions from games.

        Note: If using NFLVerseSource, drives are already extracted during parse().
        This method is for cases where you need to re-extract from raw Game objects.
        """
        possessions = []
        for game in games:
            possessions.extend(game.possessions)
        return possessions

    def compute_state_transitions(self, possessions: List[Possession]) -> List[Transition]:
        """Convert possessions into state-action-next_state tuples.

        Returns both:
        1. Play-level transitions: (state_before_play, play_type, state_after_play)
        2. Drive-level transitions: (drive_start_state, "drive", drive_end_state)
        """
        transitions = []

        for possession in possessions:
            if not possession.plays:
                continue

            # --- Play-level transitions (within-drive) ---
            if len(possession.plays) >= 2:
                for i in range(len(possession.plays) - 1):
                    current_play = possession.plays[i]
                    next_play = possession.plays[i + 1]

                    current_state = self._build_play_state(
                        current_play, possession
                    )
                    next_state = self._build_play_state(
                        next_play, possession
                    )

                    transition = Transition(
                        state=current_state,
                        action=current_play.play_type,
                        next_state=next_state,
                        team_id=possession.team,
                        opponent_id="",  # Set by caller if available
                        possession_id=possession.possession_id,
                        game_id=possession.game_id,
                        sport="nfl",
                    )
                    transitions.append(transition)

            # --- Drive-level transition (the key prediction target) ---
            first_play = possession.plays[0]
            last_play = possession.plays[-1]

            drive_start_state = self._build_drive_start_state(
                first_play, possession
            )
            drive_end_state = self._build_drive_end_state(
                last_play, possession
            )

            # Action is the drive result category
            drive_action = self._drive_result_to_action(possession.result)

            drive_transition = Transition(
                state=drive_start_state,
                action=drive_action,
                next_state=drive_end_state,
                team_id=possession.team,
                opponent_id="",  # Set by caller if available
                possession_id=possession.possession_id,
                game_id=possession.game_id,
                sport="nfl",
            )
            transitions.append(drive_transition)

        return transitions

    def _build_play_state(self, play: Play, possession: Possession) -> GameState:
        """Build a GameState from a play."""
        return GameState(
            down=play.down,
            distance=play.distance,
            yardline=play.yardline,
            score_diff=0,  # Could be enriched with game-level score
            time_remaining=0,  # Could be enriched with game-level time
            quarter=possession.quarter,
            possession=possession.team,
        )

    def _build_drive_start_state(self, first_play: Play, possession: Possession) -> GameState:
        """Build the initial state of a drive."""
        return GameState(
            down=first_play.down,
            distance=first_play.distance,
            yardline=first_play.yardline,
            score_diff=0,
            time_remaining=0,
            quarter=possession.quarter,
            possession=possession.team,
        )

    def _build_drive_end_state(self, last_play: Play, possession: Possession) -> GameState:
        """Build the terminal state of a drive."""
        return GameState(
            down=last_play.down,
            distance=last_play.distance,
            yardline=last_play.yardline,
            score_diff=possession.points_scored,  # Points gained on this drive
            time_remaining=0,
            quarter=possession.quarter,
            possession=possession.team,
        )

    def _drive_result_to_action(self, result: PossessionResult) -> str:
        """Convert a possession result to an action string for the transition model."""
        if result is None:
            return "unknown"

        mapping = {
            PossessionResult.TOUCHDOWN: "td",
            PossessionResult.FIELD_GOAL: "fg",
            PossessionResult.PUNT: "punt",
            PossessionResult.TURNOVER: "turnover",
            PossessionResult.TURNOVER_ON_DOWNS: "turnover_on_downs",
            PossessionResult.SAFETY: "safety",
            PossessionResult.END_OF_HALF: "end_of_half",
            PossessionResult.END_OF_GAME: "end_of_game",
        }
        return mapping.get(result, "unknown")

    def compute_team_stats(self, possessions: List[Possession]) -> dict:
        """Compute aggregate team statistics from possessions.

        Returns dict: team_id -> stats dict
        """
        from collections import defaultdict

        team_stats = defaultdict(lambda: {
            "total_drives": 0,
            "tds": 0,
            "field_goals": 0,
            "punts": 0,
            "turnovers": 0,
            "turnovers_on_downs": 0,
            "safeties": 0,
            "total_points": 0,
            "total_yards": 0,
            "avg_yards_per_drive": 0.0,
        })

        for possession in possessions:
            if not possession.team:
                continue

            stats = team_stats[possession.team]
            stats["total_drives"] += 1
            stats["total_points"] += possession.points_scored

            # Calculate total yards on drive
            drive_yards = sum(
                p.yards_gained for p in possession.plays
                if p.yards_gained is not None
            )
            stats["total_yards"] += drive_yards

            # Count by result
            if possession.result == PossessionResult.TOUCHDOWN:
                stats["tds"] += 1
            elif possession.result == PossessionResult.FIELD_GOAL:
                stats["field_goals"] += 1
            elif possession.result == PossessionResult.PUNT:
                stats["punts"] += 1
            elif possession.result == PossessionResult.TURNOVER:
                stats["turnovers"] += 1
            elif possession.result == PossessionResult.TURNOVER_ON_DOWNS:
                stats["turnovers_on_downs"] += 1
            elif possession.result == PossessionResult.SAFETY:
                stats["safeties"] += 1

        # Compute averages
        for team_id, stats in team_stats.items():
            if stats["total_drives"] > 0:
                stats["avg_yards_per_drive"] = (
                    stats["total_yards"] / stats["total_drives"]
                )
                stats["td_rate"] = stats["tds"] / stats["total_drives"]
                stats["fg_rate"] = stats["field_goals"] / stats["total_drives"]
                stats["punt_rate"] = stats["punts"] / stats["total_drives"]
                stats["turnover_rate"] = (
                    (stats["turnovers"] + stats["turnovers_on_downs"])
                    / stats["total_drives"]
                )

        return dict(team_stats)
