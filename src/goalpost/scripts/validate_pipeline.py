"""Validate the GoalPost data pipeline end-to-end."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from goalpost.data.nflverse_source import NFLVerseSource
from goalpost.data.nfl_drive_extractor import NFLDriveExtractor


def main():
    print("=" * 60)
    print("GoalPost Pipeline Validation")
    print("=" * 60)

    # Step 1: Fetch data
    print("\n[1/4] Fetching 2023 season data from nflverse...")
    source = NFLVerseSource()
    source.fetch(seasons=[2023])
    print(f"      ✓ Fetched {len(source._raw_data)} DataFrame(s)")

    # Step 2: Parse into domain objects
    print("\n[2/4] Parsing into Game/Possession/Play objects...")
    games = source.parse()
    print(f"      ✓ Parsed {len(games)} games")

    # Game-level sanity checks
    total_possessions = sum(len(g.possessions) for g in games)
    total_plays = sum(
        sum(len(p.plays) for p in g.possessions)
        for g in games
    )
    print(f"      ✓ Total possessions (drives): {total_possessions}")
    print(f"      ✓ Total plays: {total_plays}")

    # Check first game details
    if games:
        first_game = games[0]
        print(f"\n      Sample game: {first_game.game_id}")
        print(f"        {first_game.away_team} @ {first_game.home_team}")
        print(f"        Score: {first_game.away_score} - {first_game.home_score}")
        print(f"        Drives: {len(first_game.possessions)}")

        # Show first few possessions
        for i, poss in enumerate(first_game.possessions[:3]):
            print(f"\n        Drive {i+1}: {poss.team}")
            print(f"          Result: {poss.result.value if poss.result else 'None'}")
            print(f"          Plays: {len(poss.plays)}")
            print(f"          Points: {poss.points_scored}")
            print(f"          Start pos: {poss.start_field_position}")
            print(f"          End pos: {poss.end_field_position}")

            # Show first few plays
            for j, play in enumerate(poss.plays[:4]):
                print(f"            Play {j+1}: {play.play_type} | "
                      f"down={play.down}, dist={play.distance}, "
                      f"yardline={play.yardline}, yards={play.yards_gained}, "
                      f"pts={play.points_scored}")
            if len(poss.plays) > 4:
                print(f"            ... ({len(poss.plays) - 4} more plays)")

    # Step 3: Extract possessions
    print("\n[3/4] Extracting possessions...")
    extractor = NFLDriveExtractor()
    possessions = extractor.extract(games)
    print(f"      ✓ Extracted {len(possessions)} possessions")

    # Step 4: Compute transitions
    print("\n[4/4] Computing state transitions...")
    transitions = extractor.compute_state_transitions(possessions)
    print(f"      ✓ Computed {len(transitions)} transitions")

    # Categorize transitions
    play_level = [t for t in transitions if t.action not in [
        "td", "fg", "punt", "turnover", "turnover_on_downs",
        "safety", "end_of_half", "end_of_game", "unknown"
    ]]
    drive_level = [t for t in transitions if t.action in [
        "td", "fg", "punt", "turnover", "turnover_on_downs",
        "safety", "end_of_half", "end_of_game", "unknown"
    ]]
    print(f"      ✓ Play-level transitions: {len(play_level)}")
    print(f"      ✓ Drive-level transitions: {len(drive_level)}")

    # Step 5: Compute team stats
    print("\n[5/5] Computing team statistics...")
    team_stats = extractor.compute_team_stats(possessions)
    print(f"      ✓ Statistics for {len(team_stats)} teams")

    # Show top teams by drive count
    sorted_teams = sorted(
        team_stats.items(),
        key=lambda x: x[1]["total_drives"],
        reverse=True
    )[:5]

    print("\n      Top teams by number of drives:")
    for team_id, stats in sorted_teams:
        print(f"        {team_id}: {stats['total_drives']} drives, "
              f"{stats['tds']} TDs ({stats.get('td_rate', 0)*100:.1f}%), "
              f"{stats['punts']} punts ({stats.get('punt_rate', 0)*100:.1f}%)")

    # Sanity checks
    print("\n" + "=" * 60)
    print("Sanity Checks")
    print("=" * 60)

    # Expected: ~10-12 drives per game
    avg_drives = total_possessions / len(games) if games else 0
    print(f"Average drives per game: {avg_drives:.1f} (expected: ~22-26 total, ~11-13 per team)")

    # Expected: TD rate ~20%, punt rate ~40%
    all_stats = {
        "total_drives": sum(s["total_drives"] for s in team_stats.values()),
        "tds": sum(s["tds"] for s in team_stats.values()),
        "punts": sum(s["punts"] for s in team_stats.values()),
        "turnovers": sum(s["turnovers"] for s in team_stats.values()),
    }
    if all_stats["total_drives"] > 0:
        print(f"TD rate: {all_stats['tds']/all_stats['total_drives']*100:.1f}% (expected: ~20%)")
        print(f"Punt rate: {all_stats['punts']/all_stats['total_drives']*100:.1f}% (expected: ~40%)")
        print(f"Turnover rate: {all_stats['turnovers']/all_stats['total_drives']*100:.1f}%")

    # Check for None results
    none_results = sum(1 for p in possessions if p.result is None)
    print(f"Drives with None result: {none_results} / {len(possessions)} "
          f"({none_results/len(possessions)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
