# Purpose: Store dated transition matrices and expose historical lookup and row averages.
# Public API: HistoricalMatrix and History with add, lookup and global_aggregate methods.
# Consumers: NFLImputationBuilder uses team history and the global fallback average.

from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import numpy as np


# Purpose: Bind a transition tensor to its team, game, season and observation date.
# Consumers: History entries and NFLImputationBuilder fallback selection.
@dataclass
class HistoricalMatrix:
    date: datetime
    team_id: str
    season: int
    game_id: str
    context_count: int
    state_space_size: int
    matrix: np.ndarray  # shape (C, N, N)


# Purpose: Keep dated matrices in insertion-maintained chronological order for lookup.
# Consumers: NFLImputationBuilder and callers supplying historical matrices.
@dataclass
class History:
    entries: List[HistoricalMatrix] = field(default_factory=list)

    # Purpose: Append an observation and stably sort all entries by date.
    # Callers: History producers populating the collection before imputation.
    def add(self, entry: HistoricalMatrix) -> None:
        self.entries.append(entry)
        self.entries.sort(key=lambda x: x.date)

    # Purpose: Return matching team entries in the collection's current order.
    # Callers: NFLImputationBuilder._impute builds its team fallback candidates.
    def by_team(self, team_id: str) -> List[HistoricalMatrix]:
        return [e for e in self.entries if e.team_id == team_id]

    # Purpose: Filter a team and season without modifying the stored collection.
    # Callers: History consumers requesting a season subset; no direct repository callers.
    def by_team_season(self, team_id: str, season: int) -> List[HistoricalMatrix]:
        return [e for e in self.entries if e.team_id == team_id and e.season == season]

    # Purpose: Return the first matching team/game entry, or None when absent.
    # Callers: History consumers requesting one game; no direct repository callers.
    def by_team_game(self, team_id: str, game_id: str) -> HistoricalMatrix | None:
        for e in self.entries:
            if e.team_id == team_id and e.game_id == game_id:
                return e
        return None

    # Purpose: Average positive-sum rows from tensors with the exact requested shape.
    # Callers: NFLImputationBuilder._impute uses this final cross-team fallback.
    # Empty matches return None; rows are averaged without additional normalization.
    def global_aggregate(
        self, context: int, state: int, context_count: int, state_space_size: int
    ) -> np.ndarray | None:
        agg = np.zeros(state_space_size, dtype=np.float32)
        count = 0
        for entry in self.entries:
            m = entry.matrix
            if (
                m.shape == (context_count, state_space_size, state_space_size)
                and m[context, state].sum() > 0
            ):
                agg += m[context, state]
                count += 1
        if count == 0:
            return None
        return agg / count
