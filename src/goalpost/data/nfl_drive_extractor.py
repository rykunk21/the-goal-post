"""NFL drive extractor implementation."""

from typing import List

from ..abc.drive_extractor import DriveExtractor
from ..domain.models import Game, Drive, Play, Transition, GameState


class NFLDriveExtractor(DriveExtractor):
    """Group nflverse plays into drives and compute transitions."""

    def extract(self, games: List[Game]) -> List[Drive]:
        """Group plays by drive_id within each game."""
        # TODO: implement drive grouping
        raise NotImplementedError("extract() not yet implemented")

    def compute_state_transitions(self, drives: List[Drive]) -> List[Transition]:
        """Convert each drive into state-action-next_state tuples."""
        # TODO: implement transition extraction
        raise NotImplementedError("compute_state_transitions() not yet implemented")
