"""NFL drive extractor implementation."""

from typing import List

from ..abc.possession_extractor import PossessionExtractor
from ..domain.models import Game, Possession, Play, Transition, GameState


class NFLDriveExtractor(PossessionExtractor):
    """Group nflverse plays into drives (NFL-specific term for possessions)."""

    def extract(self, games: List[Game]) -> List[Possession]:
        """Group plays by drive_id within each game."""
        # TODO: implement drive grouping from nflverse play-by-play
        raise NotImplementedError("extract() not yet implemented")

    def compute_state_transitions(self, possessions: List[Possession]) -> List[Transition]:
        """Convert each drive into state-action-next_state tuples."""
        # TODO: implement transition extraction
        raise NotImplementedError("compute_state_transitions() not yet implemented")
