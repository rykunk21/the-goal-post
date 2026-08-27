"""NFL-specific transition model implementation.

Implements TransitionModel with NFL state spaces:
- Down/distance for play-level transitions
- Field position buckets for drive results
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

from ..abc.transition_model import TransitionModel
from ..domain.models import Game


class NFLTransitionModel(TransitionModel):
    """NFL transition matrix extracted from a SINGLE game for a SINGLE team.

    State space:
    - Play states: down_distance keys (1st_10, 2nd_long, 3rd_short, etc.)
    - Drive states: field_position buckets (red_zone, opp_40, midfield, own_40, own_20)
    - Terminal outcomes: td, fg, punt, turnover, turnover_on_downs, safety
    """

    def __init__(self, team_id: str, game_id: str, opponent_id: str):
        super().__init__(team_id, game_id, opponent_id)

        # Play transitions: (down_distance_key) -> (next_state) -> count
        self.play_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.play_totals: Dict[str, int] = defaultdict(int)

        # Drive results by field position
        self.drive_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.drive_totals: Dict[str, int] = defaultdict(int)

        # Scoring
        self.total_drives = 0
        self.total_points = 0
        self.drive_results: List[str] = []

    # NFL-specific state discretization

    def _discretize_down_distance(self, down: Optional[int], distance: Optional[int]) -> str:
        """NFL down/distance buckets."""
        if down is None:
            return "special_teams"

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
        """NFL field position buckets.

        nflverse: yardline = distance from opponent's goal line
        - 0 = opponent goal line
        - 100 = own goal line
        """
        if yardline is None:
            return "midfield"

        if yardline <= 20:
            return "red_zone"
        elif yardline <= 40:
            return "opp_40"
        elif yardline <= 60:
            return "midfield"
        elif yardline <= 80:
            return "own_40"
        else:
            return "own_20"

    # TransitionModel implementation

    def extract_from_game(self, game: Game) -> None:
        """Extract NFL transition matrix from this team's possessions."""
        team_possessions = [p for p in game.possessions if p.team == self.team_id]

        for possession in team_possessions:
            if not possession.plays:
                continue

            self.total_drives += 1
            self.total_points += possession.points_scored
            result = possession.result.value if possession.result else "unknown"
            self.drive_results.append(result)

            # Drive result by field position
            start_yardline = possession.plays[0].yardline if possession.plays else 50
            yardline_bucket = self._discretize_yardline(start_yardline)
            self.drive_counts[yardline_bucket][result] += 1
            self.drive_totals[yardline_bucket] += 1

            # Play transitions
            for i in range(len(possession.plays) - 1):
                current = possession.plays[i]
                next_play = possession.plays[i + 1]

                if current.down is None:
                    continue

                dd_key = self._discretize_down_distance(current.down, current.distance)

                if next_play.down is None:
                    next_key = result
                else:
                    next_dd = self._discretize_down_distance(next_play.down, next_play.distance)
                    next_key = next_dd

                self.play_counts[dd_key][next_key] += 1
                self.play_totals[dd_key] += 1

    def get_state_transition_probabilities(self, state_key: str) -> Dict[str, float]:
        """Get P(next_state | down_distance_key)."""
        counts = self.play_counts.get(state_key, {})
        total = self.play_totals.get(state_key, 0)

        if total == 0:
            return {}

        return {outcome: count / total for outcome, count in counts.items()}

    def simulate_drive(self, start_yardline: int = 25, max_plays: int = 20) -> Tuple[List[str], int, Optional[str]]:
        """Simulate an NFL drive using this team's transition matrix."""
        sequence = []
        points = 0

        down = 1
        distance = 10
        yardline = start_yardline

        for _ in range(max_plays):
            dd_key = self._discretize_down_distance(down, distance)
            sequence.append(dd_key)

            probs = self.get_state_transition_probabilities(dd_key)

            if not probs:
                # Fallback to drive result model
                yardline_bucket = self._discretize_yardline(yardline)
                drive_probs = self._get_drive_result_probabilities(yardline_bucket)
                if drive_probs:
                    result = self._sample(drive_probs)
                    points = self._points_from_result(result)
                    return sequence, points, result
                return sequence, 0, "punt"

            next_key = self._sample(probs)

            # Check terminal
            if next_key in ["td", "fg", "punt", "turnover", "turnover_on_downs", "safety"]:
                points = self._points_from_result(next_key)
                return sequence, points, next_key

            # Update state
            down, distance = self._parse_down_distance_key(next_key)
            if down is None:
                return sequence, 0, "punt"

            yards = self._estimate_yards(next_key)
            yardline = min(99, yardline + yards)

            if yardline >= 100:
                return sequence, 7, "td"

        return sequence, 0, "punt"

    def simulate_game(self, n_possessions: int = 12) -> int:
        """Simulate this team's NFL drives."""
        total_points = 0

        for _ in range(n_possessions):
            _, points, _ = self.simulate_drive(start_yardline=25)
            total_points += points

        return total_points

    def evaluate_reconstruction(self, actual_points: int, n_sims: int = 1000) -> Dict[str, float]:
        """Evaluate reconstruction of actual NFL game score."""
        simulated_points = []

        for _ in range(n_sims):
            points = self.simulate_game(n_possessions=self.total_drives)
            simulated_points.append(points)

        simulated_points = np.array(simulated_points)
        actual_in_distribution = np.percentile(simulated_points, 5) <= actual_points <= np.percentile(simulated_points, 95)
        prob_actual = sum(1 for p in simulated_points if p == actual_points) / n_sims

        return {
            "actual_points": actual_points,
            "mean_simulated": float(np.mean(simulated_points)),
            "median_simulated": float(np.median(simulated_points)),
            "std_simulated": float(np.std(simulated_points)),
            "min_simulated": int(np.min(simulated_points)),
            "max_simulated": int(np.max(simulated_points)),
            "prob_actual": prob_actual,
            "in_90pct_range": actual_in_distribution,
            "n_drives": self.total_drives,
        }

    def get_matrix_summary(self) -> Dict:
        """Return summary of this NFL transition matrix."""
        return {
            "team_id": self.team_id,
            "game_id": self.game_id,
            "opponent_id": self.opponent_id,
            "total_drives": self.total_drives,
            "total_points": self.total_points,
            "play_states_observed": len(self.play_counts),
            "drive_states_observed": len(self.drive_counts),
            "drive_result_distribution": {k: dict(v) for k, v in self.drive_counts.items()},
        }

    # Helper methods

    def _get_drive_result_probabilities(self, yardline_bucket: str) -> Dict[str, float]:
        """Get P(drive_result | field_position)."""
        counts = self.drive_counts.get(yardline_bucket, {})
        total = self.drive_totals.get(yardline_bucket, 0)

        if total == 0:
            return {}

        return {result: count / total for result, count in counts.items()}

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
        """Points from NFL drive result."""
        if result == "td":
            return 7
        elif result == "fg":
            return 3
        elif result == "safety":
            return 2
        return 0

    def _parse_down_distance_key(self, key: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse NFL down-distance key."""
        mapping = {
            "1st_10": (1, 10), "1st_short": (1, 5),
            "2nd_short": (2, 2), "2nd_medium": (2, 5), "2nd_long": (2, 10),
            "3rd_short": (3, 2), "3rd_medium": (3, 5), "3rd_long": (3, 10),
            "4th_short": (4, 1), "4th_medium": (4, 3), "4th_long": (4, 8),
        }
        return mapping.get(key, (None, None))

    def _estimate_yards(self, next_key: str) -> int:
        """Estimate yards for NFL outcomes."""
        mapping = {
            "1st_10": 12, "1st_short": 8,
            "2nd_short": 5, "2nd_medium": 3, "2nd_long": 0,
            "3rd_short": 3, "3rd_medium": 1, "3rd_long": 0,
            "4th_short": 2, "4th_medium": 1, "4th_long": 0,
        }
        return mapping.get(next_key, 0)


def extract_nfl_team_matrices(game: Game) -> Dict[str, NFLTransitionModel]:
    """Extract NFL transition matrices for both teams in a game.

    Returns: {team_id: NFLTransitionModel}
    """
    teams = {p.team for p in game.possessions if p.team}
    matrices = {}

    for team in teams:
        opponent = next(t for t in teams if t != team)
        matrix = NFLTransitionModel(team, game.game_id, opponent)
        matrix.extract_from_game(game)
        matrices[team] = matrix

    return matrices
