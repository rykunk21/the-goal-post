"""Drive-level transition model implementation."""

from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

from ..abc.transition_model import DriveTransitionModel
from ..domain.models import GameState, Transition, PossessionResult


class EmpiricalDriveTransitionModel(DriveTransitionModel):
    """Empirical drive result transition model.

    Predicts drive end result given starting field position and team matchup.
    This is a simpler model that just conditions on field position.
    """

    def __init__(self, latent_dim: int = 8):
        self.latent_dim = latent_dim
        # Counts: (yardline_bucket) -> (result) -> count
        self._counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._totals: Dict[str, int] = defaultdict(int)
        self._team_factors: Dict[str, Dict[str, float]] = {}

    def _discretize_yardline(self, state: GameState) -> str:
        """Convert yardline into discrete bucket.
        
        Field position is the primary driver of drive outcomes.
        """
        yardline = state.yardline or 50
        
        if yardline <= 20:
            return "own_1_20"
        elif yardline <= 40:
            return "own_21_40"
        elif yardline <= 50:
            return "midfield_41_50"
        elif yardline <= 70:
            return "opp_51_70"
        elif yardline <= 85:
            return "opp_71_85"
        else:
            return "red_zone_86_99"

    def fit(
        self,
        representations: Dict[str, np.ndarray],
        transitions: List[Transition],
    ) -> "EmpiricalDriveTransitionModel":
        """Build empirical drive result frequencies by field position."""
        for transition in transitions:
            # Only use drive-level transitions
            if transition.action not in ["td", "fg", "punt", "turnover",
                                          "turnover_on_downs", "safety",
                                          "end_of_half", "end_of_game", "unknown"]:
                continue

            state_key = self._discretize_yardline(transition.state)
            result = transition.action

            self._counts[state_key][result] += 1
            self._totals[state_key] += 1

        # Extract team factors
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
        state_key = self._discretize_yardline(state)

        # Get empirical frequencies for this field position
        counts = self._counts.get(state_key, {})
        total = self._totals.get(state_key, 0)

        if total == 0:
            return self._default_distribution(state)

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

    def _default_distribution(self, state: GameState) -> Dict[str, float]:
        """Default distribution based on field position."""
        yardline = state.yardline or 50
        
        # Field position strongly affects outcomes
        if yardline >= 86:
            # Red zone (14 yards or less)
            return {
                "td": 0.45,
                "fg": 0.25,
                "turnover": 0.15,
                "punt": 0.05,
                "turnover_on_downs": 0.08,
                "safety": 0.02,
            }
        elif yardline >= 71:
            # Opponent territory close to red zone
            return {
                "td": 0.35,
                "fg": 0.20,
                "turnover": 0.15,
                "punt": 0.15,
                "turnover_on_downs": 0.12,
                "safety": 0.03,
            }
        elif yardline >= 51:
            # Opponent territory
            return {
                "td": 0.25,
                "fg": 0.15,
                "turnover": 0.15,
                "punt": 0.30,
                "turnover_on_downs": 0.12,
                "safety": 0.03,
            }
        elif yardline >= 41:
            # Midfield
            return {
                "td": 0.18,
                "fg": 0.12,
                "turnover": 0.15,
                "punt": 0.40,
                "turnover_on_downs": 0.12,
                "safety": 0.03,
            }
        elif yardline >= 21:
            # Own territory
            return {
                "td": 0.12,
                "fg": 0.08,
                "turnover": 0.15,
                "punt": 0.50,
                "turnover_on_downs": 0.12,
                "safety": 0.03,
            }
        else:
            # Deep in own territory
            return {
                "td": 0.08,
                "fg": 0.05,
                "turnover": 0.15,
                "punt": 0.55,
                "turnover_on_downs": 0.12,
                "safety": 0.05,
            }

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
        home_td = z_home[0] if len(z_home) > 0 else 0.15
        home_fg = z_home[1] if len(z_home) > 1 else 0.10
        home_to = z_home[3] if len(z_home) > 3 else 0.15

        away_td = z_away[0] if len(z_away) > 0 else 0.15
        away_fg = z_away[1] if len(z_away) > 1 else 0.10
        away_to = z_away[3] if len(z_away) > 3 else 0.15

        # Determine offense and defense
        if possession_team == "home":
            offense_td, offense_fg = home_td, home_fg
            defense_to = away_to
        else:
            offense_td, offense_fg = away_td, away_fg
            defense_to = home_to

        # Adjust probabilities
        td_boost = offense_td / 0.20
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
