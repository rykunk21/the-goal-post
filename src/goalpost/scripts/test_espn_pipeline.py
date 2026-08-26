"""Test ESPN pipeline with the Lions @ Bengals Aug 13 2026 game."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from goalpost.data.espn_source import ESPNSource
from goalpost.data.nfl_drive_extractor import NFLDriveExtractor
from goalpost.representation.bayesian_updater import BayesianTeamUpdater


def main():
    print("=" * 60)
    print("ESPN Pipeline Test: Lions @ Bengals (Aug 13 2026)")
    print("=" * 60)

    # Step 1: Fetch from ESPN
    print("\n[1/5] Fetching game from ESPN API...")
    espn = ESPNSource()
    game = espn.load_game("401873272")
    print(f"      ✓ {game.away_team} @ {game.home_team}")
    print(f"      ✓ Score: {game.away_score} - {game.home_score}")
    print(f"      ✓ Week {game.week}, {game.season} preseason")

    # Step 2: Extract possessions
    print("\n[2/5] Extracting possessions...")
    extractor = NFLDriveExtractor()
    possessions = extractor.extract([game])
    print(f"      ✓ {len(possessions)} drives")

    # Show drive summary
    for i, drive in enumerate(possessions[:10]):
        result = drive.result.value.upper() if drive.result else "?"
        plays = len(drive.plays)
        yards = sum(p.yards_gained or 0 for p in drive.plays)
        print(f"      Drive {i+1}: {drive.team} | {result} | {plays} plays, {yards} yards, {drive.points_scored} pts")

    if len(possessions) > 10:
        print(f"      ... ({len(possessions) - 10} more drives)")

    # Step 3: Compute transitions
    print("\n[3/5] Computing transitions...")
    transitions = extractor.compute_state_transitions(possessions)
    drive_transitions = [t for t in transitions if t.action in [
        "td", "fg", "punt", "turnover", "turnover_on_downs", "safety"
    ]]
    print(f"      ✓ {len(transitions)} total transitions")
    print(f"      ✓ {len(drive_transitions)} drive-level transitions")

    # Step 4: Compute team stats
    print("\n[4/5] Computing team statistics...")
    stats = extractor.compute_team_stats(possessions)
    for team_id, team_stats in stats.items():
        print(f"\n      {team_id}:")
        print(f"        Drives: {team_stats['total_drives']}")
        print(f"        TDs: {team_stats['tds']} ({team_stats.get('td_rate', 0)*100:.1f}%)")
        print(f"        Punts: {team_stats['punts']} ({team_stats.get('punt_rate', 0)*100:.1f}%)")
        print(f"        Turnovers: {team_stats['turnovers']}")
        print(f"        Points: {team_stats['total_points']}")
        print(f"        Yards: {team_stats['total_yards']}")
        print(f"        Yards/Drive: {team_stats.get('avg_yards_per_drive', 0):.1f}")

    # Step 5: Bayesian update
    print("\n[5/5] Testing Bayesian updater...")

    # Initialize with historical priors (simulated 2025 data)
    updater = BayesianTeamUpdater(latent_dim=8)

    # Simulate fitting on historical data
    from goalpost.domain.models import PossessionResult
    historical_possessions = []

    # Create fake historical possessions for both teams
    import random
    random.seed(42)

    for team in ["DET", "CIN"]:
        for _ in range(200):  # 200 drives per team
            result = random.choices([
                PossessionResult.TOUCHDOWN,
                PossessionResult.FIELD_GOAL,
                PossessionResult.PUNT,
                PossessionResult.TURNOVER,
            ], weights=[20, 12, 40, 15])[0]

            p = type('obj', (object,), {
                'team': team,
                'result': result,
                'points_scored': 7 if result == PossessionResult.TOUCHDOWN else (3 if result == PossessionResult.FIELD_GOAL else 0),
                'plays': [],
            })()
            historical_possessions.append(p)

    updater.fit(historical_possessions)

    print("\n      Historical priors:")
    for team in ["DET", "CIN"]:
        rates = updater.get_team_rates(team)
        vec = updater.encode(team)
        print(f"        {team}: TD={rates.get('td', 0):.3f}, "
              f"Punt={rates.get('punt', 0):.3f}, "
              f"Turnover={rates.get('turnover', 0):.3f}, "
              f"conf={updater.get_confidence(team):.2f}")

    # Update with live game data
    print("\n      After live game update:")
    updater.update(game)

    for team in ["DET", "CIN"]:
        rates = updater.get_team_rates(team)
        vec = updater.encode(team)
        print(f"        {team}: TD={rates.get('td', 0):.3f}, "
              f"Punt={rates.get('punt', 0):.3f}, "
              f"Turnover={rates.get('turnover', 0):.3f}, "
              f"conf={updater.get_confidence(team):.2f}")
        print(f"        Vector: [{', '.join(f'{v:.3f}' for v in vec[:6])}]")

    print("\n" + "=" * 60)
    print("Test Complete — ESPN pipeline works for live games")
    print("=" * 60)


if __name__ == "__main__":
    main()
