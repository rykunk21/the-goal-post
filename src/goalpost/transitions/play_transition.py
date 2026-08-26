"""Play-level transition model implementation."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

from ..abc.transition_model import PlayTransitionModel
from ..domain.models import GameState, Transition, Play


class EmpiricalPlayTransitionModel(PlayTransitionModel):
    """Empirical play-to-play transition model.

    Uses observed frequencies from historical data, adjusted by team vectors.
    """

    # Yardage outcome buckets
    YARDAGE_BUCKETS = [
        "loss_or_no_gain",  # <= 0 yards
        "short_gain",         # 1-3 yards
        "medium_gain",        # 4-7 yards
        "long_gain",          # 8-15 yards
        "big_play",           # 15+ yards
    ]

    def __init__(self, latent_dim: int = 8):
        self.latent_dim = latent_dim
        self._counts: Dict = defaultdict(lambda: defaultdict(int))
        self._totals: Dict = defaultdict(int)
        self._team_offense_factors: Dict[str, np.ndarray] = {}
        self._team_defense_factors: Dict[str, np.ndarray] = {}

    def _discretize_state(self, state: GameState) -> Tuple[str, int]:
        """Convert continuous state into discrete bucket.

        Returns: (down_distance_key, yardline_bucket)
        """
        down = state.down or 1
        distance = state.distance or 10
        yardline = state.yardline or 50

        # Down & distance bucket
        if down == 1:
            dd_key = "1st_down"
        elif down == 2:
            if distance >= 8:
                dd_key = "2nd_long"
            elif distance >= 4:
                dd_key = "2nd_medium"
            else:
                dd_key = "2nd_short"
        elif down == 3:
            if distance >= 7:
                dd_key = "3rd_long"
            elif distance >= 4:
                dd_key = "3rd_medium"
            else:
                dd_key = "3rd_short"
        else:  # 4th down
            dd_key = "4th_down"

        # Yardline bucket (field position)
        if yardline <= 20:
            yard_bucket = 0  # Own red zone
        elif yardline <= 40:
            yard_bucket = 1  # Own territory
        elif yardline <= 60:
            yard_bucket = 2  # Midfield
        elif yardline <= 80:
            yard_bucket = 3  # Opp territory
        else:
            yard_bucket = 4  # Opp red zone

        return dd_key, yard_bucket

    def _yards_to_bucket(self, yards: int) -> str:
        """Convert yards gained to outcome bucket."""
        if yards <= 0:
            return "loss_or_no_gain"
        elif yards <= 3:
            return "short_gain"
        elif yards <= 7:
            return "medium_gain"
        elif yards <= 15:
            return "long_gain"
        else:
            return "big_play"

    def _bucket_to_yards(self, bucket: str) -> Tuple[int, int]:
        """Convert yardage bucket to expected yardage range.

        Returns: (min_yards, max_yards)
        """
        mapping = {
            "loss_or_no_gain": (-10, 0),
            "short_gain": (1, 3),
            "medium_gain": (4, 7),
            "long_gain": (8, 15),
            "big_play": (16, 60),
        }
        return mapping.get(bucket, (0, 0))

    def _compute_next_state(
        self, state: GameState, yards_gained: int
    ) -> GameState:
        """Compute next state after a play."""
        current_down = state.down or 1
        current_distance = state.distance or 10
        current_yardline = state.yardline or 50

        new_yardline = current_yardline + yards_gained

        # Check for touchdown
        if new_yardline >= 100:
            return GameState(
                down=None,
                distance=None,
                yardline=100,
                score_diff=state.score_diff,
                time_remaining=state.time_remaining,
                quarter=state.quarter,
                possession=state.possession,
            )

        # Check for first down
        if yards_gained >= current_distance:
            new_down = 1
            new_distance = 10
            # Adjust for red zone (distance can't exceed endzone)
            if new_yardline > 90:
                new_distance = 100 - new_yardline
        else:
            new_down = current_down + 1
            new_distance = current_distance - yards_gained
            if new_down > 4:
                # Turnover on downs — next team gets ball
                return GameState(
                    down=None,
                    distance=None,
                    yardline=new_yardline,
                    score_diff=state.score_diff,
                    time_remaining=state.time_remaining,
                    quarter=state.quarter,
                    possession=state.possession,  # Will flip in simulator
                )

        return GameState(
            down=new_down,
            distance=new_distance,
            yardline=new_yardline,
            score_diff=state.score_diff,
            time_remaining=state.time_remaining,
            quarter=state.quarter,
            possession=state.possession,
        )

    def _state_to_key(self, state: GameState) -> str:
        """Convert state to hashable key."""
        dd_key, yard_bucket = self._discretize_state(state)
        return f"{dd_key}_yb{yard_bucket}"

    def fit(
        self,
        representations: Dict[str, np.ndarray],
        transitions: List[Transition],
    ) -> "EmpiricalPlayTransitionModel":
        """Build empirical transition frequencies from play-level data."""
        for transition in transitions:
            # Only use play-level transitions (not drive results)
            if transition.action in ["td", "fg", "punt", "turnover",
                                      "turnover_on_downs", "safety",
                                      "end_of_half", "end_of_game", "unknown"]:
                continue

            state_key = self._state_to_key(transition.state)

            # Compute yards gained from state transition
            yards_gained = (
                (transition.next_state.yardline or 50) -
                (transition.state.yardline or 50)
            )

            bucket = self._yards_to_bucket(yards_gained)
            action = transition.action if transition.action else "no_play"

            # Count: P(bucket | state, action)
            self._counts[(state_key, action)][bucket] += 1
            self._totals[(state_key, action)] += 1

        # Compute team factors from representations
        for team_id, z in representations.items():
            # Offense factor: how much better than average
            self._team_offense_factors[team_id] = z[:5]  # TD, FG, Punt, TO, TOD
            # Defense factor: inverted (higher = worse defense)
            self._team_defense_factors[team_id] = 1.0 - z[:5]

        return self

    def predict(
        self,
        z_offense: np.ndarray,
        z_defense: np.ndarray,
        state: GameState,
        play_call: str,
    ) -> Dict[Tuple[Optional[int], Optional[int], Optional[int]], float]:
        """Predict next state distribution."""
        state_key = self._state_to_key(state)
        action_key = play_call if play_call else "no_play"

        # Get empirical frequencies
        counts = self._counts.get((state_key, action_key), {})
        total = self._totals.get((state_key, action_key), 0)

        if total == 0:
            # No historical data — use uniform with slight preference for short gains
            return self._default_distribution(state)

        # Build probability distribution over yardage buckets
        bucket_probs = {}
        for bucket in self.YARDAGE_BUCKETS:
            count = counts.get(bucket, 0)
            prob = count / total
            bucket_probs[bucket] = prob

        # Apply team adjustments (simplified: boost big play rate for good offenses)
        offense_boost = np.mean(z_offense[:3]) if z_offense is not None else 1.0
        defense_boost = np.mean(z_defense[:3]) if z_defense is not None else 1.0
        combined_boost = offense_boost / max(defense_boost, 0.1)

        # Adjust probabilities (boost big plays for better offenses)
        if combined_boost > 1.0:
            bucket_probs["big_play"] = min(0.5, bucket_probs.get("big_play", 0) * combined_boost)
            bucket_probs["long_gain"] = min(0.5, bucket_probs.get("long_gain", 0) * combined_boost)

        # Normalize
        total_prob = sum(bucket_probs.values())
        if total_prob > 0:
            bucket_probs = {k: v / total_prob for k, v in bucket_probs.items()}

        # Convert yardage buckets to next states
        next_states = {}
        for bucket, prob in bucket_probs.items():
            min_yards, max_yards = self._bucket_to_yards(bucket)
            # Use midpoint for deterministic state
            yards = (min_yards + max_yards) // 2
            next_state = self._compute_next_state(state, yards)
            key = (next_state.down, next_state.distance, next_state.yardline)
            next_states[key] = next_states.get(key, 0) + prob

        return next_states

    def _default_distribution(self, state: GameState) -> Dict[Tuple[Optional[int], Optional[int], Optional[int]], float]:
        """Default distribution when no historical data exists."""
        # Sensible defaults: mostly short/medium gains
        dd_key, _ = self._discretize_state(state)

        if "3rd" in dd_key:
            # Higher chance of conversion on 3rd down
            buckets = {"short_gain": 0.3, "medium_gain": 0.3, "long_gain": 0.2, "loss_or_no_gain": 0.2}
        elif "4th" in dd_key:
            # High stakes
            buckets = {"short_gain": 0.4, "loss_or_no_gain": 0.6}
        else:
            # Normal down
            buckets = {"short_gain": 0.3, "medium_gain": 0.3, "long_gain": 0.2, "big_play": 0.1, "loss_or_no_gain": 0.1}

        next_states = {}
        for bucket, prob in buckets.items():
            min_yards, max_yards = self._bucket_to_yards(bucket)
            yards = (min_yards + max_yards) // 2
            next_state = self._compute_next_state(state, yards)
            key = (next_state.down, next_state.distance, next_state.yardline)
            next_states[key] = next_states.get(key, 0) + prob

        return next_states

    def sample_play_outcome(
        self,
        z_offense: np.ndarray,
        z_defense: np.ndarray,
        state: GameState,
        play_call: str,
    ) -> GameState:
        """Sample a single next state."""
        distribution = self.predict(z_offense, z_defense, state, play_call)

        if not distribution:
            return state

        states = list(distribution.keys())
        probs = list(distribution.values())

        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            return state

        idx = np.random.choice(len(states), p=probs)
        down, distance, yardline = states[idx]

        return GameState(
            down=down,
            distance=distance,
            yardline=yardline,
            score_diff=state.score_diff,
            time_remaining=state.time_remaining,
            quarter=state.quarter,
            possession=state.possession,
        )
