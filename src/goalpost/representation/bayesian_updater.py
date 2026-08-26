"""Bayesian online updater for team representations."""

from typing import Dict, List, Optional, Any
import numpy as np
from collections import defaultdict

from ..abc.team_representation import TeamRepresentation
from ..domain.models import Game, Possession, PossessionResult


class BayesianTeamUpdater(TeamRepresentation):
    """Update team representations with new game data using Bayesian conjugate priors.

    Uses Beta-Bernoulli conjugate priors for drive outcome rates,
    which allows efficient online updates without full retraining.
    """

    # Beta(1, 1) = uniform prior as default
    DEFAULT_PRIOR_ALPHA = 1.0
    DEFAULT_PRIOR_BETA = 1.0

    def __init__(self, base_representation: Optional[Any] = None, latent_dim: int = 32):
        self.latent_dim = latent_dim
        self.base = base_representation

        # Per-team Beta posteriors for each outcome type
        # team_id -> outcome -> {"alpha": float, "beta": float, "count": int}
        self.posteriors: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {
                "alpha": self.DEFAULT_PRIOR_ALPHA,
                "beta": self.DEFAULT_PRIOR_BETA,
                "count": 0,
            })
        )

        # Per-team accumulated stats
        self.team_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_drives": 0,
            "total_points": 0,
            "total_yards": 0,
            "games_played": 0,
        })

        # Cache of computed vectors (invalidated on update)
        self._vector_cache: Dict[str, np.ndarray] = {}

        # Historical prior vector (from base representation if available)
        self._historical_priors: Dict[str, np.ndarray] = {}

    def fit(self, possessions: List[Possession]) -> "BayesianTeamUpdater":
        """Initialize from historical possessions (same interface as base encoder).

        This sets the priors based on historical data, then new games
        update the posteriors from there.
        """
        from collections import Counter

        # Group possessions by team
        team_possessions = defaultdict(list)
        for p in possessions:
            if p.team:
                team_possessions[p.team].append(p)

        # Initialize posteriors from historical frequencies
        for team_id, team_poss in team_possessions.items():
            n = len(team_poss)
            if n == 0:
                continue

            counts = Counter(p.result for p in team_poss if p.result is not None)

            # Set Beta priors based on observed historical rates
            for outcome in [PossessionResult.TOUCHDOWN, PossessionResult.FIELD_GOAL,
                           PossessionResult.PUNT, PossessionResult.TURNOVER,
                           PossessionResult.TURNOVER_ON_DOWNS, PossessionResult.SAFETY]:
                outcome_key = outcome.value
                obs_count = counts.get(outcome, 0)
                # Strong prior from historical data (pseudo-counts = 10)
                pseudo = 10.0
                rate = obs_count / n if n > 0 else 0.5
                self.posteriors[team_id][outcome_key]["alpha"] = pseudo * rate + 1.0
                self.posteriors[team_id][outcome_key]["beta"] = pseudo * (1 - rate) + 1.0
                self.posteriors[team_id][outcome_key]["count"] = obs_count

            # Set base stats
            points = sum(p.points_scored for p in team_poss)
            yards = sum(
                sum(play.yards_gained or 0 for play in p.plays)
                for p in team_poss
            )
            self.team_stats[team_id]["total_drives"] = n
            self.team_stats[team_id]["total_points"] = points
            self.team_stats[team_id]["total_yards"] = yards

            # Compute and cache initial vector
            self._vector_cache[team_id] = self._compute_vector(team_id)

        return self

    def encode(self, team_id: str, context=None) -> np.ndarray:
        """Emit a latent vector for a team.

        Returns Bayesian posterior mean vector incorporating all
        historical data + live updates so far.
        """
        if team_id not in self._vector_cache:
            self._vector_cache[team_id] = self._compute_vector(team_id)
        return self._vector_cache[team_id]

    def update(self, game: Game) -> "BayesianTeamUpdater":
        """Incorporate a new game into team beliefs via Bayesian update.

        This is the key method for live updating — call it after each
        new game is fetched from ESPN.
        """
        for possession in game.possessions:
            team = possession.team
            if not team:
                continue

            # Update posterior based on drive outcome
            result = possession.result
            if result is None:
                continue

            outcome_key = result.value

            # Beta-Bernoulli update: observe one trial with this outcome
            # Success = outcome occurred, Failure = outcome didn't occur
            # Actually better: track counts directly, then compute posterior
            self.posteriors[team][outcome_key]["alpha"] += 1.0
            self.posteriors[team][outcome_key]["count"] += 1

            # Increment "failure" counts for other outcomes
            for other_outcome in self.posteriors[team]:
                if other_outcome != outcome_key:
                    self.posteriors[team][other_outcome]["beta"] += 1.0

            # Update aggregate stats
            self.team_stats[team]["total_drives"] += 1
            self.team_stats[team]["total_points"] += possession.points_scored
            yards = sum(play.yards_gained or 0 for play in possession.plays)
            self.team_stats[team]["total_yards"] += yards

        # Mark both teams as needing recalculation
        for team in [game.home_team, game.away_team]:
            if team in self._vector_cache:
                del self._vector_cache[team]
            self.team_stats[team]["games_played"] += 1

        return self

    def batch_update(self, games: List[Game]) -> "BayesianTeamUpdater":
        """Update from multiple games efficiently."""
        for game in games:
            self.update(game)
        return self

    def _compute_vector(self, team_id: str) -> np.ndarray:
        """Compute team latent vector from current posteriors.

        Vector components (8-dimensional):
        0: P(TD) posterior mean
        1: P(FG) posterior mean
        2: P(Punt) posterior mean
        3: P(Turnover) posterior mean
        4: P(Turnover on downs) posterior mean
        5: P(Safety) posterior mean
        6: Points per drive (observed)
        7: Yards per drive (observed)
        """
        if team_id not in self.posteriors:
            # Unknown team — return neutral vector
            return np.array([0.15, 0.10, 0.40, 0.12, 0.05, 0.01, 2.0, 30.0])

        posteriors = self.posteriors[team_id]

        # Compute posterior means for each outcome
        outcome_keys = ["td", "fg", "punt", "turnover", "turnover_on_downs", "safety"]
        rates = []

        for key in outcome_keys:
            alpha = posteriors[key]["alpha"]
            beta = posteriors[key]["beta"]
            # Beta posterior mean = alpha / (alpha + beta)
            mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.0
            rates.append(mean)

        # Normalize rates to sum to ~1 (they won't exactly due to Beta smoothing)
        rate_sum = sum(rates)
        if rate_sum > 0:
            rates = [r / rate_sum for r in rates]

        # Add efficiency stats
        stats = self.team_stats[team_id]
        drives = stats["total_drives"]
        if drives > 0:
            points_per_drive = stats["total_points"] / drives
            yards_per_drive = stats["total_yards"] / drives
        else:
            points_per_drive = 0.0
            yards_per_drive = 0.0

        vector = np.array(rates + [points_per_drive, yards_per_drive], dtype=np.float32)

        # Pad or truncate to latent_dim
        if len(vector) < self.latent_dim:
            vector = np.pad(vector, (0, self.latent_dim - len(vector)), mode='constant')
        elif len(vector) > self.latent_dim:
            vector = vector[:self.latent_dim]

        return vector

    def get_team_rates(self, team_id: str) -> Dict[str, float]:
        """Get current posterior mean rates for a team (for debugging)."""
        if team_id not in self.posteriors:
            return {}

        posteriors = self.posteriors[team_id]
        rates = {}

        for outcome, params in posteriors.items():
            alpha = params["alpha"]
            beta = params["beta"]
            mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.0
            rates[outcome] = mean

        return rates

    def get_confidence(self, team_id: str) -> float:
        """Get confidence level based on observation count.

        Returns value 0-1, where 1 = high confidence (many observations).
        """
        if team_id not in self.team_stats:
            return 0.0

        drives = self.team_stats[team_id]["total_drives"]
        # Confidence grows with sqrt of observations, saturates at ~200 drives
        return min(1.0, np.sqrt(drives) / np.sqrt(200))

    def reset_to_prior(self, team_id: str) -> None:
        """Reset a team back to its prior (useful for season transitions)."""
        if team_id in self.posteriors:
            del self.posteriors[team_id]
        if team_id in self.team_stats:
            del self.team_stats[team_id]
        if team_id in self._vector_cache:
            del self._vector_cache[team_id]
