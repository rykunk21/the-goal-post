"""Raw transition matrix extraction from play-by-play data.

This module extracts transition probabilities directly from observed
frequencies in nflverse data. No modeling — just counting.

The transition matrix is the LABEL for supervised learning, not the model itself.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

from ..domain.models import GameState, Transition, PossessionResult


class TransitionMatrixExtractor:
    """Extract transition matrices from raw play-by-play data.

    Produces:
    1. Play transition matrix: P(next_down_distance | current_down_distance, play_type, yardline_bucket)
    2. Drive result matrix: P(drive_result | starting_field_position)
    """

    def __init__(self):
        # Play transitions: (down_distance_key, play_type, yardline_bucket) -> (next_down_distance_or_terminal) -> count
        self.play_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.play_totals: Dict[str, int] = defaultdict(int)

        # Drive result transitions: (yardline_bucket) -> (result) -> count
        self.drive_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.drive_totals: Dict[str, int] = defaultdict(int)

    def _discretize_down_distance(self, down: Optional[int], distance: Optional[int]) -> str:
        """Convert down/distance into discrete bucket."""
        if down is None:
            return "kickoff"

        distance = distance or 10

        if down == 1:
            return "1st_10" if distance >= 6 else "1st_short"
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
        else:  # 4th
            if distance >= 5:
                return "4th_long"
            elif distance >= 2:
                return "4th_medium"
            else:
                return "4th_short"

    def _discretize_yardline(self, yardline: Optional[int]) -> str:
        """Convert yardline into field position bucket.

        nflverse convention: yardline = distance from opponent's goal line
        - 0 = opponent goal line (touchdown)
        - 100 = own goal line
        So SMALL yardline = opponent territory, LARGE yardline = own territory
        """
        if yardline is None:
            return "midfield"

        if yardline <= 20:
            return "red_zone"  # Inside opponent 20
        elif yardline <= 40:
            return "opp_40"    # Opponent 40 to 21
        elif yardline <= 60:
            return "midfield"  # 60 to 41
        elif yardline <= 80:
            return "own_40"    # Own 40 to 21 (80 to 61)
        else:
            return "own_20"    # Deep in own territory (100 to 81)

    def extract_from_possessions(self, possessions: List) -> None:
        """Build transition matrices from observed possession data."""
        for possession in possessions:
            if not possession.plays:
                continue

            # Extract drive result
            start_yardline = possession.plays[0].yardline if possession.plays else 50
            yardline_bucket = self._discretize_yardline(start_yardline)
            result = possession.result.value if possession.result else "unknown"
            self.drive_counts[yardline_bucket][result] += 1
            self.drive_totals[yardline_bucket] += 1

            # Extract play transitions
            for i in range(len(possession.plays) - 1):
                current = possession.plays[i]
                next_play = possession.plays[i + 1]

                # Skip special teams
                if current.down is None:
                    continue

                dd_key = self._discretize_down_distance(current.down, current.distance)
                yardline_bucket = self._discretize_yardline(current.yardline)
                play_type = current.play_type if current.play_type else "unknown"

                state_key = f"{dd_key}|{play_type}|{yardline_bucket}"

                # Determine next state
                if next_play.down is None:
                    # Terminal — possession ended
                    next_key = result
                else:
                    next_dd = self._discretize_down_distance(next_play.down, next_play.distance)
                    next_key = next_dd

                self.play_counts[state_key][next_key] += 1
                self.play_totals[state_key] += 1

    def get_play_transition_probabilities(self, state_key: str) -> Dict[str, float]:
        """Get P(next_state | state_key) from observed frequencies."""
        counts = self.play_counts.get(state_key, {})
        total = self.play_totals.get(state_key, 0)

        if total == 0:
            return {}

        return {outcome: count / total for outcome, count in counts.items()}

    def get_drive_result_probabilities(self, yardline_bucket: str) -> Dict[str, float]:
        """Get P(drive_result | starting_field_position) from observed frequencies."""
        counts = self.drive_counts.get(yardline_bucket, {})
        total = self.drive_totals.get(yardline_bucket, 0)

        if total == 0:
            return {}

        return {result: count / total for result, count in counts.items()}

    def simulate_drive(self, start_yardline: int = 25, max_plays: int = 20) -> Tuple[List[str], int, Optional[str]]:
        """Simulate a single drive by walking the transition matrix.

        Returns: (sequence of states, points scored, final_result)
        """
        sequence = []
        points = 0

        # Start at 1st & 10
        down = 1
        distance = 10
        yardline = start_yardline

        for _ in range(max_plays):
            dd_key = self._discretize_down_distance(down, distance)
            yardline_bucket = self._discretize_yardline(yardline)
            play_type = self._choose_play_call(down, distance, yardline)

            state_key = f"{dd_key}|{play_type}|{yardline_bucket}"
            sequence.append(state_key)

            probs = self.get_play_transition_probabilities(state_key)

            if not probs:
                # No data — use drive model for result
                drive_probs = self.get_drive_result_probabilities(yardline_bucket)
                if drive_probs:
                    result = self._sample(drive_probs)
                    points = self._points_from_result(result)
                    return sequence, points, result
                return sequence, 0, "punt"

            # Sample next state
            next_key = self._sample(probs)

            # Check if terminal
            if next_key in ["td", "fg", "punt", "turnover", "turnover_on_downs", "safety"]:
                points = self._points_from_result(next_key)
                return sequence, points, next_key

            # Update state
            down, distance = self._parse_down_distance_key(next_key)
            if down is None:
                return sequence, 0, "punt"

            # Estimate yards gained
            yards = self._estimate_yards(next_key)
            yardline = min(99, yardline + yards)

            # Check for implicit touchdown
            if yardline >= 100:
                return sequence, 7, "td"

        # Max plays reached
        return sequence, 0, "punt"

    def simulate_game(self, n_drives: int = 24) -> Tuple[int, int]:
        """Simulate a full game by simulating drives.

        Returns: (home_score, away_score)
        """
        home_score = 0
        away_score = 0

        for i in range(n_drives):
            is_home = i % 2 == 0
            start_yardline = 25 if i > 0 else 75  # First drive after kickoff

            _, points, _ = self.simulate_drive(start_yardline)

            if is_home:
                home_score += points
            else:
                away_score += points

        return home_score, away_score

    def evaluate_against_actual(
        self,
        possessions: List,
        n_simulations: int = 1000,
    ) -> Dict[str, float]:
        """Evaluate how well the transition matrix captures actual outcomes.

        For each actual possession, simulate it many times and see if
        the actual result falls within the predicted distribution.
        """
        results = []

        for possession in possessions:
            if not possession.plays or not possession.team:
                continue

            start_yardline = possession.plays[0].yardline or 50
            actual_result = possession.result.value if possession.result else "unknown"
            actual_points = possession.points_scored

            # Simulate this drive many times
            simulated_results = defaultdict(int)
            simulated_points = []

            for _ in range(n_simulations):
                _, points, result = self.simulate_drive(start_yardline)
                simulated_results[result] += 1
                simulated_points.append(points)

            # Calculate probability of actual result
            prob_actual = simulated_results.get(actual_result, 0) / n_simulations
            prob_actual_points = sum(1 for p in simulated_points if p == actual_points) / n_simulations

            # Check if actual points is in middle 90% of distribution
            points_sorted = sorted(simulated_points)
            lower_5 = points_sorted[int(n_simulations * 0.05)]
            upper_95 = points_sorted[int(n_simulations * 0.95)]
            in_range = lower_5 <= actual_points <= upper_95

            results.append({
                "actual_result": actual_result,
                "actual_points": actual_points,
                "prob_actual": prob_actual,
                "prob_actual_points": prob_actual_points,
                "in_90pct_range": in_range,
                "mean_simulated_points": np.mean(simulated_points),
            })

        # Aggregate metrics
        avg_prob_actual = np.mean([r["prob_actual"] for r in results])
        avg_prob_points = np.mean([r["prob_actual_points"] for r in results])
        pct_in_range = sum(1 for r in results if r["in_90pct_range"]) / len(results)

        return {
            "n_possessions": len(results),
            "avg_prob_actual_result": avg_prob_actual,
            "avg_prob_actual_points": avg_prob_points,
            "pct_in_90pct_range": pct_in_range,
        }

    def _choose_play_call(self, down: int, distance: int, yardline: int) -> str:
        """Simple play-calling logic."""
        if down == 4:
            if distance <= 1 and yardline < 70:
                return "run"
            elif yardline > 60:
                return "pass"
            elif distance <= 3 and yardline > 55:
                return "pass"
            else:
                return "punt"

        if distance >= 8:
            return "pass"
        elif distance <= 2:
            return "run"
        elif yardline >= 90:
            return "run"
        else:
            return np.random.choice(["run", "pass"], p=[0.45, 0.55])

    def _sample(self, probs: Dict[str, float]) -> str:
        """Sample from probability distribution."""
        outcomes = list(probs.keys())
        probabilities = list(probs.values())

        total = sum(probabilities)
        if total == 0:
            return "punt"

        probabilities = [p / total for p in probabilities]
        return np.random.choice(outcomes, p=probabilities)

    def _points_from_result(self, result: str) -> int:
        """Points scored from drive result."""
        if result == "td":
            return 7
        elif result == "fg":
            return 3
        elif result == "safety":
            return 2
        return 0

    def _parse_down_distance_key(self, key: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse down-distance key."""
        mapping = {
            "1st_10": (1, 10),
            "1st_short": (1, 5),
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

    def _estimate_yards(self, next_key: str) -> int:
        """Estimate yards gained based on outcome."""
        mapping = {
            "1st_10": 12,
            "1st_short": 8,
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
        return mapping.get(next_key, 0)
