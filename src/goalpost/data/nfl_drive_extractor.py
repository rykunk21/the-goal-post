"""NFL drive extractor implementation."""

from typing import List

from ..abc.possession_extractor import PossessionExtractor
from ..domain.models import Game, Possession, Play, Transition, GameState


class NFLDriveExtractor(PossessionExtractor):
    """Group nflverse plays into drives (NFL-specific term for possessions)."""

    def extract(self, games: List[Game]) -> List[Possession]:
        """Group plays by drive_id within each game.
        
        Note: If using NFLVerseSource, this is already done during parse().
        This method is for cases where you need to re-extract from raw Game objects.
        """
        possessions = []
        for game in games:
            possessions.extend(game.possessions)
        return possessions

    def compute_state_transitions(self, possessions: List[Possession]) -> List[Transition]:
        """Convert each drive into state-action-next_state tuples."""
        transitions = []

        for possession in possessions:
            if len(possession.plays) < 2:
                continue

            for i in range(len(possession.plays) - 1):
                current_play = possession.plays[i]
                next_play = possession.plays[i + 1]

                # Build current state
                current_state = GameState(
                    down=current_play.down,
                    distance=current_play.distance,
                    yardline=current_play.yardline,
                    score_diff=0,  # Would need game-level score tracking
                    time_remaining=0,  # Would need time parsing
                    quarter=possession.quarter,
                    possession=possession.team,
                )

                # Build next state
                next_state = GameState(
                    down=next_play.down,
                    distance=next_play.distance,
                    yardline=next_play.yardline,
                    score_diff=0,
                    time_remaining=0,
                    quarter=possession.quarter,
                    possession=possession.team,
                )

                transition = Transition(
                    state=current_state,
                    action=current_play.play_type,
                    next_state=next_state,
                    team_id=possession.team,
                    opponent_id="",  # Would need game context
                    possession_id=possession.possession_id,
                    game_id=possession.game_id,
                    sport="nfl",
                )
                transitions.append(transition)

        return transitions
