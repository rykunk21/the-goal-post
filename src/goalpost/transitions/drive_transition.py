"""Drive-level transition model implementation."""

from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

from ..abc.transition_model import DriveTransitionModel
from ..domain.models import GameState, Transition, PossessionResult


class EmpiricalDriveTransitionModel(DriveTransitionModel):
    """Empirical drive result transition model.

    Predicts drive end result given starting state and team matchup.
    """

    def __init__(self, latent_dim: int = 8):
        self.latent_dim = latent_dim
        self._counts: Dict = defaultdict(lambda: defaultdict(int))
        self._totals: Dict = defaultdict(int)
        self._team_factors: Dict[str, Dict[str, float]] = {}

    def _discretize_start_state(self, state: GameState) -> str:
        """Convert drive start state into discrete bucket.

        Key factors: field position, score differential, time/quarter.
        """
        yardline = state.yardline or 25
        score_diff = state.score_diff
        quarter = state.quarter or 1

        # Field position bucket
        if yardline <= 20:
            field_pos = "own_red_zone"
        elif yardline <= 40:
            field_pos = "own_territory"
        elif yardline <= 60:
            field_pos = "midfield"
        elif yardline <= 80:
            field_pos = "opp_territory"
        else:
            field_pos = "opp_red_zone"

        # Score differential bucket
        if score_diff <= -14:
            score_bucket = "down_big"
        elif score_diff <= -7:
            score_bucket = "down_td"
        elif score_diff < 0:
            score_bucket = "down_small"
        elif score_diff == 0:
            score_bucket = "tied"
        elif score_diff <= 7:
            score_bucket = "up_small"
        elif score_diff <= 14:
            score_bucket = "up_td"
        else:
            score_bucket = "up_big"

        # Time/quarter
        if quarter >= 4:
            time_bucket = "late"
        elif quarter >= 3:
            time_bucket = "mid_late"
        else:
            time_bucket = "early"

        return f"{field_pos}_{score_bucket}_{time_bucket}"

    def fit(
        self,
        representations: Dict[str, np.ndarray],
        transitions: List[Transition],
    ) -> "EmpiricalDriveTransitionModel":
        """Build empirical drive result frequencies."""
        for transition in transitions:
            # Only use drive-level transitions
            if transition.action not in ["td", "fg", "punt", "turnover",
                                          "turnover_on_downs", "safety",
                                          "end_of_half", "end_of_game", "unknown"]:
                continue

            state_key = self._discretize_start_state(transition.state)
            result = transition.action

            self._counts[state_key][result] += 1
            self._totals[state_key] += 1

        # Extract team factors from representations
        for team_id, z in representations.items():
            self._team_factors[team_id] = {
                "td_rate": z[0] if len(z) > 0 else 0.15,
                "fg_rate": z[1] if len(z) > 1 else 0.10,
                "punt_rate": z[2] if len(z) > 2 else 0.40,
                "turnover_rate": z[3] if len(z) > 3 else 0.15,
            }

        return self

    def predict(
        self,
        z_home: np.ndarray,
        z_away: np.ndarray,
        state: GameState,
    ) -> Dict[str, float]:
        """Predict drive result probabilities.

        Returns: {result: probability}
        Results: td, fg, punt, turnover, turnover_on_downs, safety
        """
        state_key = self._discretize_start_state(state)

        # Get empirical frequencies
        counts = self._counts.get(state_key, {})
        total = self._totals.get(state_key, 0)

        if total == 0:
            # No historical data for this state — use defaults
            return self._default_distribution(state, z_home, z_away)

        # Build base distribution
        base_probs = {}
        for result in ["td", "fg", "punt", "turnover", "turnover_on_downs", "safety"]:
            count = counts.get(result, 0)
            base_probs[result] = count / total

        # Apply team adjustments
        adjusted = self._apply_team_factors(
            base_probs, z_home, z_away, state.possession
        )

        # Normalize
        total_prob = sum(adjusted.values())
        if total_prob > 0:
            adjusted = {k: float(v) / float(total_prob) for k, v in adjusted.items()}

        return adjusted

    def _default_distribution(
        self, state: GameState, z_home: np.ndarray, z_away: np.ndarray
    ) -> Dict[str, float]:
        """Default distribution when no historical data."""
        yardline = state.yardline or 50

        # Field position strongly affects outcomes
        if yardline >= 80:
            # Opp red zone: high TD chance
            base = {"td": 0.45, "fg": 0.25, "turnover": 0.15, "punt": 0.05, "turnover_on_downs": 0.05, "safety": 0.05}
        elif yardline >= 60:
            # Opp territory
            base = {"td": 0.30, "fg": 0.20, "turnover": 0.15, "punt": 0.25, "turnover_on_downs": 0.05, "safety": 0.05}
        elif yardline >= 40:
            # Midfield
            base = {"td": 0.20, "fg": 0.15, "turnover": 0.15, "punt": 0.40, "turnover_on_downs": 0.05, "safety": 0.05}
        else:
            # Own territory
            base = {"td": 0.10, "fg": 0.10, "turnover": 0.15, "punt": 0.55, "turnover_on_downs": 0.05, "safety": 0.05}

        return self._apply_team_factors(base, z_home, z_away, state.possession)

    def _apply_team_factors(
        self,
        base_probs: Dict[str, float],
        z_home: np.ndarray,
        z_away: np.ndarray,
        possession_team: str,
    ) -> Dict[str, float]:
        """Adjust base probabilities by team matchup."""
        adjusted = dict(base_probs)

        # Extract team-specific rates from vectors
        # z format: [td_rate, fg_rate, punt_rate, turnover_rate, turnover_on_downs, safety, pts/drive, yards/drive]
        home_td = z_home[0] if len(z_home) > 0 else 0.15
        home_fg = z_home[1] if len(z_home) > 1 else 0.10
        home_to = z_home[3] if len(z_home) > 3 else 0.15

        away_td = z_away[0] if len(z_away) > 0 else 0.15
        away_fg = z_away[1] if len(z_away) > 1 else 0.10
        away_to = z_away[3] if len(z_away) > 3 else 0.15

        # Determine offense and defense
        if possession_team == "home":
            offense_td, offense_fg = home_td, home_fg
            defense_to = away_to  # Opponent's turnover rate = defense's takeaway tendency
        else:
            offense_td, offense_fg = away_td, away_fg
            defense_to = home_to

        # Adjust probabilities
        # Better offense = higher TD, lower punt
        td_boost = offense_td / 0.20  # Normalize around 20% baseline
        fg_boost = offense_fg / 0.12
        to_boost = defense_to / 0.15

        adjusted["td"] = min(0.8, adjusted.get("td", 0) * td_boost)
        adjusted["fg"] = min(0.6, adjusted.get("fg", 0) * fg_boost)
        adjusted["turnover"] = min(0.5, adjusted.get("turnover", 0) * to_boost)

        # Ensure probabilities sum to reasonable range (renormalize later)
        return adjusted

    def sample_drive_result(
        self,
        z_home: np.ndarray,
        z_away: np.ndarray,
        state: GameState,
    ) -> Tuple[str, int]:
        """Sample a drive result.

        Returns: (result, points_scored)
        """
        distribution = self.predict(z_home, z_away, state)

        if not distribution:
            return "punt", 0

        results = list(distribution.keys())
        probs = list(distribution.values())

        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            return "punt", 0

        idx = np.random.choice(len(results), p=probs)
        result = results[idx]

        # Determine points
        points = 0
        if result == "td":
            points = 7  # Including XP
        elif result == "fg":
            points = 3
        elif result == "safety":
            points = 2

        return result, points
