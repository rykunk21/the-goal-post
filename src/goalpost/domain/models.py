"""Domain models for GoalPost."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import numpy as np


class PossessionResult(Enum):
    """Possible end states of a possession. Sport-specific subclasses extend this."""
    # NFL / CFB
    TOUCHDOWN = "td"
    FIELD_GOAL = "fg"
    PUNT = "punt"
    TURNOVER = "turnover"
    TURNOVER_ON_DOWNS = "turnover_on_downs"
    SAFETY = "safety"
    # Universal
    END_OF_HALF = "end_of_half"
    END_OF_GAME = "end_of_game"
    # NBA
    MADE_BASKET = "made_basket"
    # MLB
    THREE_OUTS = "three_outs"
    # Hockey / Soccer
    GOAL = "goal"


@dataclass
class Play:
    """A single play / action."""
    play_id: str
    # NFL-specific (nullable for other sports)
    down: Optional[int] = None  # 1-4, null for NBA/MLB
    distance: Optional[int] = None  # yards to go, null for NBA/MLB
    yardline: Optional[int] = None  # 1-100, distance from opponent endzone
    # Universal
    play_type: str  # pass, run, shot, at_bat, etc.
    yards_gained: Optional[int] = None  # null for NBA (points instead)
    points_scored: int = 0  # 0 for most plays, 2/3/6/7 for scores
    epa: float = 0.0  # expected points added (sport-specific model)
    wp: float = 0.5
    passer: Optional[str] = None
    rusher: Optional[str] = None
    receiver: Optional[str] = None
    penalty: bool = False
    turnover: bool = False
    scoring_play: bool = False


@dataclass
class Possession:
    """A contiguous sequence of plays where one team controls the ball.
    
    Called a 'drive' in NFL, 'possession' in NBA, 'inning half' in MLB, etc.
    """
    possession_id: str
    team: str
    plays: List[Play] = field(default_factory=list)
    result: Optional[PossessionResult] = None
    quarter: int = 1
    start_field_position: Optional[int] = None  # sport-specific
    end_field_position: Optional[int] = None
    points_scored: int = 0
    game_id: Optional[str] = None
    sport: str = "nfl"


@dataclass
class Game:
    """A single game with all possessions."""
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    possessions: List[Possession] = field(default_factory=list)
    home_score: int = 0
    away_score: int = 0
    date: Optional[str] = None
    sport: str = "nfl"


@dataclass
class GameState:
    """State representation for transition modeling.
    
    Fields are sport-specific where applicable (nullable for non-matching sports).
    """
    # NFL / CFB / continuous-field sports
    down: Optional[int] = None
    distance: Optional[int] = None
    yardline: Optional[int] = None  # field position (1-100)
    # NBA / court sports
    shot_clock: Optional[int] = None
    # Universal
    score_diff: int = 0  # from possession team's perspective
    time_remaining: int = 0  # seconds in quarter/half/game
    quarter: int = 1
    period: int = 1  # synonym, for hockey/baseball
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
    drive_id: Optional[str] = None  # backward compat alias
    possession_id: Optional[str] = None
    game_id: Optional[str] = None
    sport: str = "nfl"


@dataclass
class GameOutcome:
    """Result of a simulated game."""
    home_score: int = 0
    away_score: int = 0
    total: int = 0
    margin: int = 0
    first_half_margin: int = 0
    home_possessions: int = 0  # drives, innings, etc.
    away_possessions: int = 0
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
