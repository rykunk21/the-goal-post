"""Test the hybrid transition model and simulator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from goalpost.data.nflverse_source import NFLVerseSource
from goalpost.data.nfl_drive_extractor import NFLDriveExtractor
from goalpost.representation.bayesian_updater import BayesianTeamUpdater
from goalpost.transitions.play_transition import EmpiricalPlayTransitionModel
from goalpost.transitions.drive_transition import EmpiricalDriveTransitionModel
from goalpost.simulator.game_simulator import MonteCarloSimulator


def main():
    print("=" * 60)
    print("Hybrid Transition Model Test")
    print("=" * 60)

    # Step 1: Load historical data
    print("\n[1/6] Loading 2023 season data...")
    source = NFLVerseSource()
    source.fetch(seasons=[2023])
    games = source.parse()
    print(f"      ✓ {len(games)} games loaded")

    # Step 2: Extract possessions and transitions
    print("\n[2/6] Extracting possessions and transitions...")
    extractor = NFLDriveExtractor()
    all_possessions = []
    for game in games:
        all_possessions.extend(game.possessions)

    # Get play-level and drive-level transitions
    all_transitions = extractor.compute_state_transitions(all_possessions)
    play_transitions = [t for t in all_transitions if t.action not in [
        "td", "fg", "punt", "turnover", "turnover_on_downs",
        "safety", "end_of_half", "end_of_game", "unknown"
    ]]
    drive_transitions = [t for t in all_transitions if t.action in [
        "td", "fg", "punt", "turnover", "turnover_on_downs",
        "safety", "end_of_half", "end_of_game", "unknown"
    ]]

    print(f"      ✓ {len(all_possessions)} possessions")
    print(f"      ✓ {len(play_transitions)} play-level transitions")
    print(f"      ✓ {len(drive_transitions)} drive-level transitions")

    # Step 3: Build team representations
    print("\n[3/6] Building Bayesian team representations...")
    updater = BayesianTeamUpdater(latent_dim=8)
    updater.fit(all_possessions)

    # Get team vectors for a few teams
    teams = ["KC", "SF", "DET", "BAL"]
    team_vectors = {}
    for team in teams:
        vec = updater.encode(team)
        team_vectors[team] = vec
        print(f"      {team}: {vec[:5]}")

    # Step 4: Fit play transition model
    print("\n[4/6] Fitting play transition model...")
    play_model = EmpiricalPlayTransitionModel(latent_dim=8)
    play_model.fit(team_vectors, play_transitions)
    print(f"      ✓ States observed: {len(play_model._counts)}")

    # Test play prediction
    from goalpost.domain.models import GameState
    test_state = GameState(down=1, distance=10, yardline=25)
    pred = play_model.predict(team_vectors["KC"], team_vectors["SF"], test_state, "pass")
    print(f"      Sample prediction (1st & 10, own 25, KC pass vs SF):")
    for (down, dist, yardline), prob in sorted(pred.items(), key=lambda x: -x[1])[:5]:
        print(f"        -> {down}d {dist} @ {yardline}: {prob:.3f}")

    # Step 5: Fit drive transition model
    print("\n[5/6] Fitting drive transition model...")
    drive_model = EmpiricalDriveTransitionModel(latent_dim=8)
    drive_model.fit(team_vectors, drive_transitions)
    print(f"      ✓ States observed: {len(drive_model._counts)}")

    # Test drive prediction
    test_state2 = GameState(down=1, distance=10, yardline=25, quarter=1)
    pred2 = drive_model.predict(team_vectors["KC"], team_vectors["SF"], test_state2)
    print(f"      Sample prediction (drive start, KC vs SF):")
    for result, prob in sorted(pred2.items(), key=lambda x: -x[1]):
        print(f"        {result}: {prob:.3f}")

    # Step 6: Simulate games
    print("\n[6/6] Running Monte Carlo simulations...")
    simulator = MonteCarloSimulator(play_model, drive_model)

    outcomes = simulator.simulate_game(
        z_home=team_vectors["KC"],
        z_away=team_vectors["SF"],
        home_team="KC",
        away_team="SF",
        n_sims=1000,
    )

    markets = simulator.price_markets(outcomes)
    print(f"\n      KC vs SF Simulated Markets (1000 sims):")
    print(f"        Spread: {markets['spread'][0]:.1f} (KC wins {markets['spread'][1]*100:.1f}%)")
    print(f"        Total: {markets['total'][0]:.1f}")
    print(f"        KC ML: {markets['moneyline_home'][1]*100:.1f}%")
    print(f"        SF ML: {markets['moneyline_away'][1]*100:.1f}%")

    # Distribution
    margins = [o.margin for o in outcomes]
    totals = [o.total for o in outcomes]
    print(f"\n      Margin distribution:")
    print(f"        Mean: {np.mean(margins):.1f}")
    print(f"        Std: {np.std(margins):.1f}")
    print(f"        Median: {np.median(margins):.1f}")
    print(f"        KC wins by 7+: {sum(1 for m in margins if m >= 7)/len(margins)*100:.1f}%")
    print(f"        SF wins by 7+: {sum(1 for m in margins if m <= -7)/len(margins)*100:.1f}%")

    print(f"\n      Total distribution:")
    print(f"        Mean: {np.mean(totals):.1f}")
    print(f"        Median: {np.median(totals):.1f}")
    print(f"        Over 42.5: {sum(1 for t in totals if t > 42.5)/len(totals)*100:.1f}%")

    print("\n" + "=" * 60)
    print("Hybrid Model Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
