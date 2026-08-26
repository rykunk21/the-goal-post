"""Play-level transition model implementation."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

from ..abc.transition_model import PlayTransitionModel
from ..domain.models import GameState, Transition, Play, PossessionResult


class EmpiricalPlayTransitionModel(PlayTransitionModel):
    """Empirical play-to-play transition model.

    Predicts what happens on a given down/distance.
    Next state is either a new down/distance OR a terminal outcome.
    """

    # Terminal outcomes that end a possession
    TERMINAL_OUTCOMES = ["td", "fg", "punt", "turnover", "turnover_on_downs", "safety"]

    def __init__(self, latent_dim: int = 8):
        self.latent_dim = latent_dim
        # Counts: (down_distance_key) -> (next_down_distance_or_terminal) -> count
        self._counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._totals: Dict[str, int] = defaultdict(int)
        self._team_factors: Dict[str, Dict[str, float]] = {}

    def _discretize_down_distance(self, state: GameState) -> str:
        """Convert down/distance into discrete bucket.

        Yardline is NOT part of play state — it's handled by drive model.
        """
        down = state.down or 1
        distance = state.distance or 10

        if down == 1:
            # 1st down: distance matters less (always 10 except red zone)
            if distance <= 5:
                return "1st_short"
            else:
                return "1st_long"
        elif down == 2:
            if distance >= 8:
                return "2nd_long"
            elif distance >= 4:
                return "2nd_medium"
            else:
                return "2nd_short"
        elif down == 3:
            if distance >= 7:
                return "3rd_long"
            elif distance >= 4:
                return "3rd_medium"
            else:
                return "3rd_short"
        else:  # 4th down
            if distance >= 5:
                return "4th_long"
            elif distance >= 2:
                return "4th_medium"
            else:
                return "4th_short"

    def fit(
        self,
        representations: Dict[str, np.ndarray],
        transitions: List[Transition],
    ) -> "EmpiricalPlayTransitionModel":
        """Build empirical transition frequencies from play-level data."""
        for transition in transitions:
            # Skip if this IS a terminal outcome (those are drive transitions)
            if transition.action in self.TERMINAL_OUTCOMES:
                continue

            state_key = self._discretize_down_distance(transition.state)

            # Determine next state
            next_state = transition.next_state
            if next_state.down is None or next_state.down > 4:
                # Terminal — what kind?
                if hasattr(transition, 'possession_result') and transition.possession_result:
                    next_key = transition.possession_result.value
                else:
                    # Infer from action or next state
                    next_key = self._infer_terminal_outcome(transition)
            else:
                # New down/distance
                next_key = self._discretize_down_distance(next_state)

            self._counts[state_key][next_key] += 1
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

    def _infer_terminal_outcome(self, transition: Transition) -> str:
        """Infer terminal outcome from transition data."""
        # Check if yards gained suggests score
        yards = (transition.next_state.yardline or 50) - (transition.state.yardline or 50)
        if yards >= 50:
            return "td"
        elif transition.action in ["punt", "punt_attempt"]:
            return "punt"
        elif transition.action in ["field_goal", "field_goal_attempt"]:
            return "fg"
        elif "interception" in transition.action or "fumble" in transition.action:
            return "turnover"
        elif transition.action in ["safety"]:
            return "safety"
        else:
            return "punt"  # Default

    def predict(
        self,
        z_offense: np.ndarray,
        z_defense: np.ndarray,
        state: GameState,
        play_call: str,
    ) -> Dict[str, float]:
        """Predict next down/distance OR terminal outcome.

        Returns: {outcome: probability}
        Outcomes are either down-distance keys ("2nd_short", "3rd_long", etc.)
        or terminal outcomes ("td", "fg", "punt", "turnover")
        """
        state_key = self._discretize_down_distance(state)

        # Get empirical frequencies
        counts = self._counts.get(state_key, {})
        total = self._totals.get(state_key, 0)

        if total == 0:
            return self._default_distribution(state)

        # Build probability distribution
        probs = {}
        for outcome, count in counts.items():
            probs[outcome] = count / total

        # Apply team adjustments
        # Better offense = more conversions, fewer punts
        offense_td = z_offense[0] if len(z_offense) > 0 else 0.15
        offense_punt = z_offense[2] if len(z_offense) > 2 else 0.40
        defense_td = z_defense[0] if len(z_defense) > 0 else 0.15

        td_boost = (offense_td / 0.20) * (defense_td / 0.20)
        punt_suppress = (0.40 / max(offense_punt, 0.1))

        # Adjust terminal outcomes
        for outcome in ["td", "fg"]:
            if outcome in probs:
                probs[outcome] = min(0.8, probs[outcome] * td_boost)
        if "punt" in probs:
            probs["punt"] = probs["punt"] * punt_suppress

        # Normalize
        total_prob = sum(probs.values())
        if total_prob > 0:
            probs = {k: float(v) / float(total_prob) for k, v in probs.items()}

        return probs

    def _default_distribution(self, state: GameState) -> Dict[str, float]:
        """Default distribution when no historical data."""
        down = state.down or 1
        distance = state.distance or 10

        if down == 1:
            # 1st down: usually get something, rarely terminal
            return {
                "2nd_short": 0.35,
                "2nd_medium": 0.30,
                "2nd_long": 0.20,
                "1st_long": 0.10,
                "punt": 0.05,
            }
        elif down == 2:
            if distance >= 8:
                return {"3rd_long": 0.45, "3rd_medium": 0.25, "1st_long": 0.15, "punt": 0.10, "td": 0.05}
            elif distance >= 4:
                return {"3rd_medium": 0.35, "3rd_short": 0.25, "1st_medium": 0.20, "punt": 0.15, "td": 0.05}
            else:
                return {"3rd_short": 0.30, "1st_short": 0.35, "punt": 0.25, "td": 0.10}
        elif down == 3:
            if distance >= 7:
                return {"4th_long": 0.50, "punt": 0.30, "turnover": 0.15, "td": 0.05}
            elif distance >= 4:
                return {"4th_medium": 0.35, "punt": 0.30, "1st_medium": 0.20, "turnover": 0.10, "td": 0.05}
            else:
                return {"4th_short": 0.25, "1st_short": 0.30, "turnover": 0.20, "td": 0.20, "punt": 0.05}
        else:  # 4th down
            if distance >= 5:
                return {"punt": 0.60, "turnover": 0.25, "turnover_on_downs": 0.10, "fg": 0.05}
            elif distance >= 2:
                return {"punt": 0.40, "fg": 0.25, "turnover": 0.20, "turnover_on_downs": 0.10, "td": 0.05}
            else:
                return {"fg": 0.35, "td": 0.30, "turnover": 0.20, "turnover_on_downs": 0.15}

    def sample_play_outcome(
        self,
        z_offense: np.ndarray,
        z_defense: np.ndarray,
        state: GameState,
        play_call: str,
    ) -> Tuple[Optional[GameState], Optional[str]]:
        """Sample a single next state.

        Returns: (next_state, terminal_outcome)
        If terminal_outcome is not None, the possession ended.
        """
        distribution = self.predict(z_offense, z_defense, state, play_call)

        if not distribution:
            return None, "punt"

        outcomes = list(distribution.keys())
        probs = list(distribution.values())

        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            return None, "punt"

        idx = np.random.choice(len(outcomes), p=probs)
        outcome = outcomes[idx]

        # Check if terminal
        if outcome in self.TERMINAL_OUTCOMES:
            return None, outcome

        # Parse down-distance key
        down, distance = self._parse_down_distance_key(outcome)
        if down is None:
            return None, "punt"

        # Compute new yardline (simplified — actual yardline depends on yards gained)
        current_yardline = state.yardline or 50
        # Estimate yards gained based on outcome
        yards = self._estimate_yards_gained(outcome)
        new_yardline = min(99, current_yardline + yards)

        next_state = GameState(
            down=down,
            distance=distance,
            yardline=new_yardline,
            score_diff=state.score_diff,
            time_remaining=state.time_remaining,
            quarter=state.quarter,
            possession=state.possession,
        )

        return next_state, None

    def _parse_down_distance_key(self, key: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse a down-distance key like '2nd_short' into (down, distance)."""
        mapping = {
            "1st_short": (1, 5),
            "1st_long": (1, 10),
            "2nd_short": (2, 2),
            "2nd_medium": (2, 5),
            "2nd_long": (2, 10),
            "3rd_short": (3, 2),
            "3rd_medium": (3, 5),
            "3rd_long": (3, 10),
            "4th_short": (4, 1),
            "4th_medium": (4, 3),
            "4th_long": (4, 8),
        }
        return mapping.get(key, (None, None))

    def _estimate_yards_gained(self, outcome: str) -> int:
        """Estimate yards gained from outcome key."""
        mapping = {
            "1st_short": 8,
            "1st_long": 12,
            "2nd_short": 5,
            "2nd_medium": 3,
            "2nd_long": 0,
            "3rd_short": 3,
            "3rd_medium": 1,
            "3rd_long": 0,
            "4th_short": 2,
            "4th_medium": 1,
            "4th_long": 0,
        }
        return mapping.get(outcome, 0)
