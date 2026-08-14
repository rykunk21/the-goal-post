"""Domain models for GoalPost."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import numpy as np


class DriveResult(Enum):
    """Possible end states of a drive."""
    TOUCHDOWN = "td"
    FIELD_GOAL = "fg"
    PUNT = "punt"
    TURNOVER = "turnover"
    TURNOVER_ON_DOWNS = "turnover_on_downs"
    END_OF_HALF = "end_of_half"
    END_OF_GAME = "end_of_game"
    SAFETY = "safety"


@dataclass
class Play:
    """A single play."""
    play_id: str
    down: int
    distance: int
    yardline: int  # 1-100, distance from opponent endzone
    play_type: str  # pass, run, penalty, etc.
    yards_gained: int
    epa: float = 0.0
    wp: float = 0.5
    passer: Optional[str] = None
    rusher: Optional[str] = None
    receiver: Optional[str] = None
    penalty: bool = False
    turnover: bool = False
    scoring_play: bool = False


@dataclass
class Drive:
    """A sequence of plays ending in a drive result."""
    drive_id: str
    team: str
    plays: List[Play] = field(default_factory=list)
    result: Optional[DriveResult] = None
    quarter: int = 1
    start_yardline: int = 25
    end_yardline: Optional[int] = None
    points_scored: int = 0
    game_id: Optional[str] = None


@dataclass
class Game:
    """A single game with all drives."""
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    drives: List[Drive] = field(default_factory=list)
    home_score: int = 0
    away_score: int = 0
    date: Optional[str] = None


@dataclass
class GameState:
    """State representation for transition modeling."""
    down: int
    distance: int
    yardline: int
    score_diff: int  # from possession team's perspective
    time_remaining: int  # seconds in quarter or game
    quarter: int = 1
    possession: str = ""  # team with ball
    timeouts_offense: int = 3
    timeouts_defense: int = 3


@dataclass
class GameContext:
    """Context for encoding a team representation."""
    season: int
    week: int
    opponent_id: str
    is_home: bool
    game_type: str = "REG"  # REG or POST


@dataclass
class Transition:
    """A state-action-next_state tuple for learning."""
    state: GameState
    action: str  # play category or outcome bucket
    next_state: GameState
    team_id: str
    opponent_id: str
    drive_id: Optional[str] = None
    game_id: Optional[str] = None


@dataclass
class GameOutcome:
    """Result of a simulated game."""
    home_score: int = 0
    away_score: int = 0
    total: int = 0
    margin: int = 0
    first_half_margin: int = 0
    home_drives: int = 0
    away_drives: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Market:
    """A betting market to price."""
    market_key: str  # e.g. "spreads", "totals", "h2h"
    side: str  # e.g. "home", "away", "over", "under"
    line: float = 0.0  # for spreads/totals
    odds: float = 0.0  # American odds


# Type alias for transition matrix
TransitionMatrix = Dict[str, Dict[str, float]]  # from_state -> {to_state: prob}
