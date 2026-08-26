"""Game simulator using hybrid transition model."""

from typing import List, Dict, Tuple, Optional
import numpy as np

from ..transitions.play_transition import EmpiricalPlayTransitionModel
from ..transitions.drive_transition import EmpiricalDriveTransitionModel
from ..domain.models import GameState, GameOutcome, GameContext


class MonteCarloSimulator:
    """Simulate NFL games using play-level and drive-level transitions."""

    def __init__(
        self,
        play_model: Optional[EmpiricalPlayTransitionModel] = None,
        drive_model: Optional[EmpiricalDriveTransitionModel] = None,
    ):
        self.play_model = play_model
        self.drive_model = drive_model

    def simulate_game(
        self,
        z_home: np.ndarray,
        z_away: np.ndarray,
        home_team: str,
        away_team: str,
        n_sims: int = 1000,
    ) -> List[GameOutcome]:
        """Run Monte Carlo simulations of a game.

        Returns: List of GameOutcome objects (one per simulation)
        """
        outcomes = []
        for _ in range(n_sims):
            outcome = self._simulate_single_game(z_home, z_away, home_team, away_team)
            outcomes.append(outcome)
        return outcomes

    def _simulate_single_game(
        self,
        z_home: np.ndarray,
        z_away: np.ndarray,
        home_team: str,
        away_team: str,
    ) -> GameOutcome:
        """Simulate a single game."""
        home_score = 0
        away_score = 0
        home_possessions = 0
        away_possessions = 0

        # Typical NFL game: ~22-24 drives total
        # Alternate possessions
        teams = [home_team, away_team]
        current_team_idx = 0

        for _ in range(24):  # Max drives
            team = teams[current_team_idx]
            is_home = (team == home_team)
            z_offense = z_home if is_home else z_away
            z_defense = z_away if is_home else z_home

            # Start drive from own 25 (typical kickoff result)
            state = GameState(
                down=1,
                distance=10,
                yardline=25,
                score_diff=home_score - away_score if is_home else away_score - home_score,
                time_remaining=3600,  # Full game (simplified)
                quarter=1,
                possession=team,
            )

            # Simulate drive
            drive_points = self._simulate_drive(
                z_offense, z_defense, state, is_home
            )

            # Update score
            if is_home:
                home_score += drive_points
                home_possessions += 1
            else:
                away_score += drive_points
                away_possessions += 1

            # Switch possession
            current_team_idx = 1 - current_team_idx

        return GameOutcome(
            home_score=home_score,
            away_score=away_score,
            total=home_score + away_score,
            margin=home_score - away_score,
            home_possessions=home_possessions,
            away_possessions=away_possessions,
        )

    def _simulate_drive(
        self,
        z_offense: np.ndarray,
        z_defense: np.ndarray,
        start_state: GameState,
        is_home: bool,
    ) -> int:
        """Simulate a single drive.

        Returns: points scored (0, 3, or 7)
        """
        state = GameState(
            down=start_state.down,
            distance=start_state.distance,
            yardline=start_state.yardline,
            score_diff=start_state.score_diff,
            time_remaining=start_state.time_remaining,
            quarter=start_state.quarter,
            possession=start_state.possession,
        )

        max_plays = 15  # Prevent infinite loops
        for _ in range(max_plays):
            # Check for drive end conditions
            if state.yardline is not None and state.yardline >= 100:
                # Touchdown
                return 7

            if state.down is None or state.down > 4:
                # Turnover on downs or other end
                if self.drive_model:
                    result, points = self.drive_model.sample_drive_result(
                        z_offense if is_home else z_defense,
                        z_defense if is_home else z_offense,
                        state,
                    )
                    return points
                return 0

            # Decide play call (simplified)
            play_call = self._choose_play_call(state)

            # Simulate play
            if self.play_model:
                next_states = self.play_model.predict(
                    z_offense, z_defense, state, play_call
                )

                if next_states:
                    states = list(next_states.keys())
                    probs = list(next_states.values())

                    # Normalize
                    total = sum(probs)
                    if total > 0:
                        probs = [p / total for p in probs]
                        idx = np.random.choice(len(states), p=probs)
                        down, distance, yardline = states[idx]

                        state = GameState(
                            down=down,
                            distance=distance,
                            yardline=yardline,
                            score_diff=state.score_diff,
                            time_remaining=state.time_remaining,
                            quarter=state.quarter,
                            possession=state.possession,
                        )
                    else:
                        break
                else:
                    break
            else:
                # No play model — use drive model directly
                break

        # If we exited the loop, use drive model for result
        if self.drive_model:
            result, points = self.drive_model.sample_drive_result(
                z_offense if is_home else z_defense,
                z_defense if is_home else z_offense,
                state,
            )
            return points

        return 0

    def _choose_play_call(self, state: GameState) -> str:
        """Simple play-calling strategy."""
        down = state.down or 1
        distance = state.distance or 10
        yardline = state.yardline or 50

        # 4th down decisions
        if down == 4:
            if distance <= 1 and yardline < 70:
                return "run"
            elif distance <= 3 and yardline > 65:
                return "pass"
            elif yardline > 60:
                return "field_goal"
            else:
                return "punt"

        # Normal downs
        if distance >= 8:
            return "pass"
        elif distance <= 2:
            return "run"
        elif yardline >= 90:
            return "run"  # Red zone
        else:
            # Balanced
            return np.random.choice(["run", "pass"], p=[0.45, 0.55])

    def price_markets(
        self,
        outcomes: List[GameOutcome],
    ) -> Dict[str, Tuple[float, float]]:
        """Price betting markets from simulation outcomes.

        Returns: {market: (line, probability)}
        """
        margins = [o.margin for o in outcomes]
        totals = [o.total for o in outcomes]

        # Spread
        mean_margin = np.mean(margins)
        median_margin = np.median(margins)

        # Total
        mean_total = np.mean(totals)
        median_total = np.median(totals)

        # Win probability
        home_wins = sum(1 for o in outcomes if o.margin > 0)
        away_wins = sum(1 for o in outcomes if o.margin < 0)
        ties = sum(1 for o in outcomes if o.margin == 0)
        n = len(outcomes)

        return {
            "spread": (mean_margin, home_wins / n),
            "total": (mean_total, 0.5),  # Simplified
            "moneyline_home": (0.0, home_wins / n),
            "moneyline_away": (0.0, away_wins / n),
        }
