"""NFL transition model — single simulatable transition set.

One matrix: play-level Markov chain that captures everything needed
to simulate a game. No separate drive-result matrix.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

from ..abc.transition_model import TransitionModel
from ..domain.models import Game


class NFLTransitionModel(TransitionModel):
    """NFL transition matrix extracted from a SINGLE game for a SINGLE team.

    A single simulatable transition set: play-level transitions where
    terminal outcomes are just another state transition.

    States are (down, distance, yardline_bucket). Terminal outcomes
    are absorbing states with points attached.
    """

    # Full state space — every state the model can be in
    DOWNS = [1, 2, 3, 4]
    DISTANCE_BUCKETS = ["short", "medium", "long"]  # short ≤ 3, medium ≤ 7, long > 7
    YARDLINE_BUCKETS = ["own_20", "own_40", "midfield", "opp_40", "red_zone", "goal_line"]

    TERMINAL_OUTCOMES = ["td", "fg", "punt", "turnover", "turnover_on_downs", "safety"]

    @classmethod
    def state_space(cls) -> List[str]:
        """Return ordered list of all state keys for flatten/unflatten."""
        states = []
        for yardline in cls.YARDLINE_BUCKETS:
            for down in cls.DOWNS:
                for distance in cls.DISTANCE_BUCKETS:
                    states.append(f"{down}d_{distance}_{yardline}")
        states.extend(cls.TERMINAL_OUTCOMES)
        return states

    @classmethod
    def input_dim(cls) -> int:
        """Size of flattened probability vector."""
        n_play_states = len(cls.DOWNS) * len(cls.DISTANCE_BUCKETS) * len(cls.YARDLINE_BUCKETS)
        n_terminal = len(cls.TERMINAL_OUTCOMES)
        total_states = n_play_states + n_terminal
        # Each state transitions to terminal outcomes
        return total_states * n_terminal

    def __init__(self, team_id: str, game_id: str, opponent_id: str):
        super().__init__(team_id, game_id, opponent_id)

        # Single transition matrix: from_state -> to_state -> count
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.totals: Dict[str, int] = defaultdict(int)

        # Scoring
        self.total_drives = 0
        self.total_points = 0

    # NFL-specific state discretization

    @staticmethod
    def _discretize_distance(distance: Optional[int]) -> str:
        """Short/medium/long buckets."""
        if distance is None:
            return "long"
        if distance <= 3:
            return "short"
        if distance <= 7:
            return "medium"
        return "long"

    @staticmethod
    def _discretize_yardline(yardline: Optional[int]) -> str:
        """Field position buckets.

        nflverse: yardline = distance from opponent's goal line
        - 0 = opponent goal line
        - 100 = own goal line
        """
        if yardline is None:
            return "midfield"

        if yardline <= 5:
            return "goal_line"
        if yardline <= 20:
            return "red_zone"
        if yardline <= 40:
            return "opp_40"
        if yardline <= 60:
            return "midfield"
        if yardline <= 80:
            return "own_40"
        return "own_20"

    def _build_state_key(self, down: int, distance: int, yardline: int) -> str:
        """Build a state key from raw down/distance/yardline."""
        return (
            f"{down}d_"
            f"{self._discretize_distance(distance)}_"
            f"{self._discretize_yardline(yardline)}"
        )

    # TransitionModel implementation

    def extract_from_game(self, game: Game) -> None:
        """Extract single transition matrix from this team's possessions."""
        team_possessions = [p for p in game.possessions if p.team == self.team_id]

        for possession in team_possessions:
            if not possession.plays:
                continue

            self.total_drives += 1
            self.total_points += possession.points_scored

            # Track each play transition within the possession
            plays = possession.plays
            for i in range(len(plays) - 1):
                current = plays[i]
                next_play = plays[i + 1]

                from_state = self._build_state_key(
                    current.down or 1,
                    current.distance or 10,
                    current.yardline or 50,
                )

                # Determine next state
                if next_play.down is None:
                    # Terminal: use the drive result
                    result = possession.result.value if possession.result else "unknown"
                    to_state = self._result_to_terminal(result)
                else:
                    to_state = self._build_state_key(
                        next_play.down,
                        next_play.distance or 10,
                        next_play.yardline or 50,
                    )

                self.counts[from_state][to_state] += 1
                self.totals[from_state] += 1

            # Also capture the initial state of the drive
            # (so we know where drives started)
            first_play = plays[0]
            start_state = self._build_state_key(
                first_play.down or 1,
                first_play.distance or 10,
                first_play.yardline or 50,
            )
            self.counts["__start__"][start_state] += 1
            self.totals["__start__"] += 1

    def _result_to_terminal(self, result: str) -> str:
        """Map possession result to terminal state key."""
        mapping = {
            "TOUCHDOWN": "td",
            "FIELD_GOAL": "fg",
            "PUNT": "punt",
            "TURNOVER": "turnover",
            "TURNOVER_ON_DOWNS": "turnover_on_downs",
            "SAFETY": "safety",
        }
        return mapping.get(result, "turnover")

    def get_state_transition_probabilities(self, state_key: str) -> Dict[str, float]:
        """Get P(next_state | state_key)."""
        counts = self.counts.get(state_key, {})
        total = self.totals.get(state_key, 0)

        if total == 0:
            return {}

        return {outcome: count / total for outcome, count in counts.items()}

    def simulate_drive(self, start_yardline: int = 25, max_plays: int = 20) -> Tuple[List[str], int, Optional[str]]:
        """Simulate an NFL drive using this team's transition matrix.

        Returns: (state_sequence, points_scored, terminal_outcome)
        """
        sequence = []
        points = 0

        down = 1
        distance = 10
        yardline = start_yardline

        # Start state
        current = f"{down}d_{self._discretize_distance(distance)}_{self._discretize_yardline(yardline)}"

        for _ in range(max_plays):
            sequence.append(current)

            probs = self.get_state_transition_probabilities(current)

            if not probs:
                # No data for this state — fallback to nearest observed state
                probs = self._fallback_probabilities(current)
                if not probs:
                    return sequence, 0, "punt"

            next_state = self._sample(probs)

            # Check if terminal
            if next_state in self.TERMINAL_OUTCOMES:
                points = self._points_from_result(next_state)
                return sequence, points, next_state

            # Update state from next_state key
            current = next_state

            # Parse down/distance/yardline from state key
            parsed = self._parse_state_key(current)
            if parsed is None:
                return sequence, 0, "punt"

            down, distance, yardline = parsed

            # Check for touchdown via yardline
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
        n_play_states = len(self.DOWNS) * len(self.DISTANCE_BUCKETS) * len(self.YARDLINE_BUCKETS)
        return {
            "team_id": self.team_id,
            "game_id": self.game_id,
            "opponent_id": self.opponent_id,
            "total_drives": self.total_drives,
            "total_points": self.total_points,
            "states_observed": len(self.counts),
            "total_possible_states": n_play_states + len(self.TERMINAL_OUTCOMES),
            "terminal_distribution": {
                k: dict(v) for k, v in self.counts.items()
                if k in self.TERMINAL_OUTCOMES or any(
                    outcome in self.TERMINAL_OUTCOMES for outcome in v.keys()
                )
            },
        }

    # Flatten / unflatten for encoder/decoder

    def flatten_probabilities(self) -> List[float]:
        """Flatten transition probabilities into a fixed-size vector.

        Order: for each state in state_space(), probability of each terminal outcome.
        Unobserved transitions are 0.0.
        """
        flat = []
        states = self.state_space()
        for state in states:
            probs = self.get_state_transition_probabilities(state)
            for outcome in self.TERMINAL_OUTCOMES:
                flat.append(probs.get(outcome, 0.0))
        return flat

    @classmethod
    def from_flat_probabilities(
        cls, probs: List[float], team_id: str, game_id: str, opponent_id: str
    ) -> "NFLTransitionModel":
        """Reconstruct an NFLTransitionModel from a flattened probability vector.

        Used at inference time when the decoder outputs predicted probabilities.
        """
        model = cls(team_id, game_id, opponent_id)
        states = cls.state_space()
        idx = 0
        for state in states:
            for outcome in cls.TERMINAL_OUTCOMES:
                if idx < len(probs) and probs[idx] > 0:
                    model.counts[state][outcome] = int(probs[idx] * 1000)
                    model.totals[state] += int(probs[idx] * 1000)
                idx += 1
        return model

    # Helper methods

    def _fallback_probabilities(self, state_key: str) -> Dict[str, float]:
        """Find nearest observed state when exact state has no data."""
        # Try ignoring yardline bucket
        parts = state_key.split("_")
        if len(parts) >= 3:
            down_dist = f"{parts[0]}_{parts[1]}"
            # Search for any state with same down/distance
            for observed_state in self.counts:
                if observed_state.startswith(down_dist):
                    return self.get_state_transition_probabilities(observed_state)
        return {}

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

    def _parse_state_key(self, key: str) -> Optional[Tuple[int, int, int]]:
        """Parse a state key into (down, distance, yardline).

        State keys look like: "1d_short_own_20", "3d_long_red_zone"
        """
        if key in self.TERMINAL_OUTCOMES:
            return None

        try:
            # Split: "1d_short_own_20" -> parts
            parts = key.split("_")
            down = int(parts[0][0])

            # Distance bucket -> nominal distance
            dist_map = {"short": 2, "medium": 5, "long": 10}
            distance = dist_map.get(parts[1], 10)

            # Yardline bucket -> nominal yardline
            yardline_map = {
                "own_20": 80, "own_40": 60, "midfield": 50,
                "opp_40": 40, "red_zone": 15, "goal_line": 5,
            }
            # The yardline is parts[2] + "_" + parts[3] if it has two parts
            if len(parts) >= 4:
                yardline_key = f"{parts[2]}_{parts[3]}"
            else:
                yardline_key = parts[2]
            yardline = yardline_map.get(yardline_key, 50)

            return down, distance, yardline
        except (ValueError, IndexError):
            return None


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
