"""Test single-game transition matrix extraction and evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from goalpost.data.nflverse_source import NFLVerseSource
from goalpost.transitions.single_game_matrix import extract_team_matrices, SingleGameTransitionMatrix


def test_single_game():
    print("=" * 60)
    print("Single Game Transition Matrix Test")
    print("=" * 60)

    # Load ONE specific game
    print("\n[1/3] Loading 2023 Week 1 KC vs DET...")
    source = NFLVerseSource()
    source.fetch(seasons=[2023])
    games = source.parse()

    # Find KC vs DET Week 1
    target_game = None
    for g in games:
        if g.week == 1:
            if (g.home_team == "KC" and g.away_team == "DET") or (g.home_team == "DET" and g.away_team == "KC"):
                target_game = g
                break

    if not target_game:
        # Fallback to first game
        target_game = games[0]

    print(f"      Game: {target_game.away_team} @ {target_game.home_team}")
    print(f"      Score: {target_game.away_score} - {target_game.home_score}")
    print(f"      Drives: {len(target_game.possessions)}")

    # Extract transition matrices for BOTH teams
    print("\n[2/3] Extracting transition matrices...")
    matrices = extract_team_matrices(target_game)

    for team_id, matrix in matrices.items():
        print(f"\n      Team: {team_id}")
        print(f"      Opponent: {matrix.opponent_id}")
        print(f"      Drives: {matrix.total_drives}")
        print(f"      Points: {matrix.total_points}")
        print(f"      Play states observed: {len(matrix.play_counts)}")
        print(f"      Drive states observed: {len(matrix.drive_counts)}")

        # Show play transitions
        print(f"\n      Play transitions:")
        for dd_key in sorted(matrix.play_totals.keys(), key=lambda k: matrix.play_totals[k], reverse=True):
            total = matrix.play_totals[dd_key]
            print(f"        {dd_key}: {total} plays")
            outcomes = matrix.play_counts[dd_key]
            for outcome, count in sorted(outcomes.items(), key=lambda x: -x[1])[:3]:
                print(f"          -> {outcome}: {count}/{total} ({count/total:.2f})")

        # Show drive results
        print(f"\n      Drive results by field position:")
        for yardline_bucket in sorted(matrix.drive_totals.keys()):
            total = matrix.drive_totals[yardline_bucket]
            print(f"        {yardline_bucket}: {total} drives")
            results = matrix.drive_counts[yardline_bucket]
            for result, count in sorted(results.items(), key=lambda x: -x[1]):
                print(f"          {result}: {count}/{total} ({count/total:.2f})")

    # Evaluate reconstruction
    print("\n[3/3] Evaluating score reconstruction...")

    for team_id, matrix in matrices.items():
        # Get actual points for this team
        actual_points = target_game.home_score if team_id == target_game.home_team else target_game.away_score

        print(f"\n      {team_id}: Actual = {actual_points} points")
        print(f"      Simulating 1000 times using {team_id}'s transition matrix...")

        # Simulate using this team's matrix
        result = matrix.evaluate_reconstruction(actual_points, n_sims=1000)

        print(f"\n      Results:")
        print(f"        Actual points: {result['actual_points']}")
        print(f"        Mean simulated: {result['mean_simulated']:.1f}")
        print(f"        Median simulated: {result['median_simulated']:.1f}")
        print(f"        Std dev: {result['std_simulated']:.1f}")
        print(f"        Range: {result['min_simulated']}-{result['max_simulated']}")
        print(f"        P(actual score): {result['prob_actual']:.3f}")
        print(f"        In 90% range: {result['in_90pct_range']}")

        # Show distribution
        import numpy as np
        simulated = []
        for _ in range(1000):
            points = matrix.simulate_game(n_drives=matrix.total_drives)
            simulated.append(points)

        # Count frequency of actual score
        actual_count = simulated.count(actual_points)
        print(f"\n        Distribution around actual ({actual_points} pts):")
        for pts in range(max(0, actual_points-7), actual_points+8):
            count = simulated.count(pts)
            marker = " ***" if pts == actual_points else ""
            bar = "█" * int(count / 5)
            print(f"          {pts:2d} pts: {count:3d} {bar}{marker}")

    print("\n" + "=" * 60)
    print("Single Game Matrix Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_single_game()
