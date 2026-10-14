# Purpose: represent provider-independent plays, possessions and games as typed records.
# Public API: Play, Possession and Game dataclasses; defaults permit partial source data.
# Consumers: NFLDataSource constructs these records and data.base re-exports them.
# These containers do not validate identity or infer scores from their nested records.

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


# Purpose: retain one source play's identity, category and supplied scoring value.
# Consumers: NFLDataSource.parse creates plays inside Possession.plays.
# A zero scoring default is a placeholder unless a source explicitly populates it.
@dataclass
class Play:
    play_id: str = ""
    play_type: str = ""
    points_scored: int = 0


# Purpose: group a team's plays with optional source possession and game identifiers.
# Consumers: NFLDataSource.parse creates possessions inside Game.possessions.
# Each instance owns its plays list; points_scored is not computed from that list.
@dataclass
class Possession:
    possession_id: str = ""
    team: str = ""
    plays: List[Play] = field(default_factory=list)
    points_scored: int = 0
    game_id: Optional[str] = None


# Purpose: carry matchup identity with optional detail for historical or upcoming games.
# Consumers: NFLDataSource.parse and fetch_upcoming return Game records to source clients.
# Upcoming records may retain empty possessions and zero score defaults, not final results.
# date retains the source datetime; this record does not enforce a timezone convention.
@dataclass
class Game:
    game_id: str = ""
    season: int = 0
    week: int = 0
    home_team: str = ""
    away_team: str = ""
    possessions: List[Possession] = field(default_factory=list)
    home_score: int = 0
    away_score: int = 0
    date: Optional[datetime] = None
