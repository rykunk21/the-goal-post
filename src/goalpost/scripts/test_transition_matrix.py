"""Test transition matrix extraction and evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from goalpost.data.nflverse_source import NFLVerseSource
from goalpost.transitions.transition_matrix import TransitionMatrixExtractor


def main():
    print("=" * 60)
    print("Transition Matrix Extraction Test")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading 2023 season data...")
    source = NFLVerseSource()
    source.fetch(seasons=[2023])
    games = source.parse()
    print(f"      ✓ {len(games)} games loaded")

    # Split: train on weeks 1-14, test on weeks 15-18
    train_games = [g for g in games if g.week <= 14]
    test_games = [g for g in games if g.week > 14]

    print(f"      Train: {len(train_games)} games (weeks 1-14)")
    print(f"      Test:  {len(test_games)} games (weeks 15-18)")

    # Extract possessions
    train_possessions = []
    for g in train_games:
        train_possessions.extend(g.possessions)

    test_possessions = []
    for g in test_games:
        test_possessions.extend(g.possessions)

    print(f"      Train possessions: {len(train_possessions)}")
    print(f"      Test possessions:  {len(test_possessions)}")

    # Build transition matrix from train data
    print("\n[2/4] Building transition matrices...")
    extractor = TransitionMatrixExtractor()
    extractor.extract_from_possessions(train_possessions)

    print(f"      ✓ Play transition states: {len(extractor.play_counts)}")
    print(f"      ✓ Drive result states: {len(extractor.drive_counts)}")

    # Show most common play transitions
    print("\n      Most common play transitions:")
    for state_key in sorted(extractor.play_totals.keys(), key=lambda k: extractor.play_totals[k], reverse=True)[:5]:
        total = extractor.play_totals[state_key]
        print(f"        {state_key}: {total} observations")
        outcomes = extractor.play_counts[state_key]
        for outcome, count in sorted(outcomes.items(), key=lambda x: -x[1])[:3]:
            print(f"          -> {outcome}: {count}/{total} ({count/total:.2f})")

    # Show drive result by field position
    print("\n      Drive results by field position:")
    for yardline_bucket in sorted(extractor.drive_totals.keys()):
        total = extractor.drive_totals[yardline_bucket]
        print(f"        {yardline_bucket}: {total} drives")
        results = extractor.drive_counts[yardline_bucket]
        for result, count in sorted(results.items(), key=lambda x: -x[1]):
            print(f"          {result}: {count}/{total} ({count/total:.2f})")

    # Simulate some drives
    print("\n[3/4] Simulating individual drives...")
    for start_pos, name in [(25, "own 25"), (50, "midfield"), (75, "opp 25")]:
        simulated_results = {}
        simulated_points = []

        for _ in range(1000):
            _, points, result = extractor.simulate_drive(start_pos)
            simulated_results[result] = simulated_results.get(result, 0) + 1
            simulated_points.append(points)

        import numpy as np
        print(f"\n      Drive from {name} (1000 sims):")
        for result, count in sorted(simulated_results.items(), key=lambda x: -x[1]):
            print(f"        {result}: {count/10:.1f}%")
        print(f"        Avg points: {np.mean(simulated_points):.2f}")
        print(f"        Points distribution: 0pts={sum(1 for p in simulated_points if p==0)/10:.1f}%, 3pts={sum(1 for p in simulated_points if p==3)/10:.1f}%, 7pts={sum(1 for p in simulated_points if p==7)/10:.1f}%")

    # Evaluate against actual test data
    print("\n[4/4] Evaluating against actual test data...")
    metrics = extractor.evaluate_against_actual(test_possessions, n_simulations=100)

    print(f"\n      Evaluation Metrics:")
    print(f"        Possessions evaluated: {metrics['n_possessions']}")
    print(f"        Avg P(actual result): {metrics['avg_prob_actual_result']:.3f}")
    print(f"        Avg P(actual points): {metrics['avg_prob_actual_points']:.3f}")
    print(f"        % in 90% range: {metrics['pct_in_90pct_range']*100:.1f}%")

    # Show individual examples
    print("\n      Sample evaluations:")
    for possession in test_possessions[:5]:
        if not possession.plays or not possession.team:
            continue

        start_yardline = possession.plays[0].yardline or 50
        actual_result = possession.result.value if possession.result else "unknown"
        actual_points = possession.points_scored

        # Simulate this drive
        sim_results = {}
        sim_points = []
        for _ in range(100):
            _, points, result = extractor.simulate_drive(start_yardline)
            sim_results[result] = sim_results.get(result, 0) + 1
            sim_points.append(points)

        prob_actual = sim_results.get(actual_result, 0) / 100
        points_in_range = sum(1 for p in sim_points if p == actual_points) / 100

        print(f"\n        {possession.team} from {start_yardline}: actual={actual_result} ({actual_points}pts)")
        print(f"          P(actual result) = {prob_actual:.2f}")
        print(f"          P(actual points) = {points_in_range:.2f}")
        print(f"          Simulated points: mean={np.mean(sim_points):.1f}, "
              f"min={min(sim_points)}, max={max(sim_points)}")

    print("\n" + "=" * 60)
    print("Transition Matrix Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
