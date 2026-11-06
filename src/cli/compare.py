"""One-click simulation-to-market comparison CLI.

Orchestration flow:
1. Discover or create a weekly simulation export.
2. Load the export as NFLGame objects with PMFs.
3. Fetch live market quotes via Odds API.
4. Run the pricing comparison.
5. Print a clean stdout table (no root-level CSVs).

All artifacts are written under artifacts/market-sizing/<timestamp>/.
"""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from markets.comparison import ComparisonSettings, MarketRow, NFLComparison
from markets.export import WeeklyExport
from markets.history import NFLHistory
from markets.quotes import OddsAPISource, QuoteSnapshot
from markets.nfl import NFLGame, NFLSpread
from transitions.weekly import main as weekly_main
from transitions.paths import ARTIFACTS

import numpy as np
import pandas as pd


def _find_latest_export() -> Path | None:
    """Return the most recent valid weekly simulation export directory."""
    weekly_dir = ARTIFACTS / "weekly"
    if not weekly_dir.exists():
        return None
    candidates = [
        d for d in weekly_dir.iterdir()
        if d.is_dir() and (d / "report.json").exists()
    ]
    if not candidates:
        return None
    # Sort by directory name (ISO timestamps) descending
    candidates.sort(key=lambda p: p.name, reverse=True)
    for cand in candidates:
        try:
            report = json.loads((cand / "report.json").read_text())
            if report.get("status") in ("complete", "incomplete"):
                return cand
        except (ValueError, OSError):
            continue
    return None


def _run_simulations(
    season: int | None = None,
    week: int | None = None,
    simulations: int = 50000,
) -> Path:
    """Delegate to the weekly simulation runner and return the export path."""
    import io
    import sys

    argv = ["--simulations", str(simulations)]
    if season is not None:
        argv.extend(["--season", str(season)])
    if week is not None:
        argv.extend(["--week", str(week)])

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        rc = weekly_main(argv)
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    finally:
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

    if rc != 0:
        raise RuntimeError(f"Weekly simulation failed (exit {rc})")
    parsed = json.loads(output.strip().splitlines()[-1])
    return Path(parsed["output"])


def _median(distribution) -> float:
    """Median of a ScoreDistribution."""
    cum = np.cumsum(distribution.probabilities)
    idx = int(np.argmin(np.abs(cum - 0.5)))
    return float(distribution.points[idx])


def _print_comparison_table(current_rows: list[dict], games: list[NFLGame]) -> None:
    """Pretty-print the fresh-quote comparison table to stdout."""
    if not current_rows:
        print("No fresh quotes to display.")
        return

    game_map = {g.game_id: g for g in games}

    # Group rows by game
    games_data = {}
    for row in current_rows:
        gid = row["game_id"]
        if gid not in games_data:
            games_data[gid] = {
                "home": row["home"],
                "away": row["away"],
                "spread_rows": [],
                "total_rows": [],
            }
        games_data[gid][f"{row['market']}_rows"].append(row)

    print("\n" + "=" * 110)
    print("NFL SIMULATION vs MARKET COMPARISON")
    print("=" * 110)
    print(
        f"{'Game':<25}{'Market Spread':<18}{'Sim Median':<14}{'P(Home Cover)':<16}{'P(Away Cover)':<16}{'Edge (Home %)':<14}{'Sim Total':<12}"
    )
    print("-" * 110)

    for gid, info in games_data.items():
        game = game_map.get(gid)
        spread_rows = info["spread_rows"]

        home_row = next((r for r in spread_rows if r["participant"] == info["home"]), None)
        if home_row is None:
            continue

        p_home = home_row.get("p_win", 0.0)
        p_away = home_row.get("p_loss", 0.0)
        edge = (p_home - p_away) * 100
        market_spread = home_row["line"]

        sim_med = "N/A"
        sim_total = "N/A"
        if game is not None:
            try:
                sim_med = f"{_median(game.home_margin):+.1f}"
            except Exception:
                pass
            try:
                sim_total = f"{_median(game.total):.1f}"
            except Exception:
                pass

        game_str = f"{info['away']} @ {info['home']}"
        market = f"{market_spread:+.1f}" if market_spread is not None else "N/A"
        p_home_str = f"{p_home:.2%}"
        p_away_str = f"{p_away:.2%}"
        edge_str = f"{edge:+.1f}%"

        print(
            f"{game_str:<25}{market:<18}{sim_med:<14}{p_home_str:<16}{p_away_str:<16}{edge_str:<14}{sim_total:<12}"
        )

    print("-" * 110)
    print("INTERPRETATION:")
    print("  - Positive Edge(Home): Simulation favors home team more than market")
    print("  - Negative Edge(Home): Simulation favors away team more than market")
    print("  - Higher absolute edge = larger disagreement between model and market")
    print("=" * 110)


def run_compare(
    export_path: Path | None = None,
    season: int | None = None,
    week: int | None = None,
    simulations: int = 50000,
    book: str = "fanduel",
    bankroll: float | None = None,
    max_quote_age_minutes: float = 30,
    kickoff_tolerance_seconds: float = 600,
) -> dict:
    """End-to-end compare: simulate if needed, fetch odds, run comparison, print table."""
    # --- Step 1: Ensure we have a simulation export ---
    if export_path is None:
        export_path = _find_latest_export()
        if export_path is None:
            print("No existing simulation export found. Running simulations...")
            export_path = _run_simulations(season=season, week=week, simulations=simulations)
            print(f"Simulation export written to: {export_path}\n")
        else:
            print(f"Using existing simulation export: {export_path}\n")

    # --- Step 2: Load the export ---
    export = WeeklyExport.load(export_path)
    now = datetime.now(timezone.utc)

    # --- Step 3: History check ---
    history = NFLHistory().check(export.report, now)
    if history["status"] == "blocked":
        raise RuntimeError("Simulation history is incomplete; inspect history.json")

    # --- Step 4: Fetch live quotes ---
    if any(game.kickoff > now for game in export.games):
        source = OddsAPISource(book)
        snapshot = source.load()
    else:
        snapshot = QuoteSnapshot((), "not_requested")

    now = datetime.now(timezone.utc)

    # --- Step 5: Run comparison ---
    settings = ComparisonSettings(
        book=book,
        max_quote_age_seconds=max_quote_age_minutes * 60,
        kickoff_tolerance_seconds=kickoff_tolerance_seconds,
        bankroll=bankroll,
    )
    comparison = NFLComparison(settings).run(export.games, snapshot, now)
    rows, current = comparison["rows"], comparison["current"]

    # --- Step 6: Write artifacts under artifacts/market-sizing/ ---
    output_dir = ARTIFACTS / "market-sizing" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    columns = [field.name for field in fields(MarketRow)]
    for name, values in (("ev-kelly", rows), ("current-market-sizing", current)):
        (output_dir / f"{name}.csv").write_text(_to_csv(values, columns))
        (output_dir / f"{name}.json").write_text(
            json.dumps(values, indent=2, allow_nan=False) + "\n"
        )

    (output_dir / "quote-evidence.json").write_text(
        json.dumps(comparison["quote_evidence"], indent=2, allow_nan=False) + "\n"
    )

    matchups = comparison["matchups"] + export.unavailable
    summary = {
        "status": "complete",
        "as_of_utc": now.isoformat(),
        "source": snapshot.source,
        "book": book,
        "settings": asdict(settings),
        "games": len(matchups),
        "attempted": sum(game["attempted"] for game in matchups),
        "retained": sum(game["retained"] for game in matchups),
        "approximate": sum(game["approximate"] for game in matchups),
        "evaluated_rows": len(rows),
        "fresh_rows": len(current),
        "quote_status_counts": dict(Counter(row["quote_status"] for row in rows)),
        "odds_requests": snapshot.requests,
        "credits_last_request": snapshot.credits_last_request,
        "credits_used": snapshot.credits_used,
        "credits_remaining": snapshot.credits_remaining,
        "history": history,
        "calculation_scope": "hypothetical_regulation",
        "limitations": [
            "Uncalibrated regulation model; actual sportsbook EV/Kelly remains unset.",
            "Mass uses retained simulations; missing paths can bias probabilities.",
            "Regulation moneyline ties are treated as void, not modeled through overtime.",
            "Stale and invalid rows are retained but excluded from the current table.",
            "Correlated single-contract Kelly fractions must not be added together.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )

    # --- Step 7: Print table ---
    _print_comparison_table(current, export.games)
    print(f"\nArtifacts written to: {output_dir}")
    return summary


def _to_csv(rows: List[dict], columns: List[str]) -> str:
    import csv
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="One-click simulation-to-market comparison. Runs simulations if none exist."
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Use an existing weekly simulation export directory instead of discovering/running one",
    )
    parser.add_argument("--season", type=int, help="NFL season year")
    parser.add_argument("--week", type=int, help="NFL week number")
    parser.add_argument(
        "--simulations", type=int, default=50000, help="Simulations per game (default 50000)"
    )
    parser.add_argument("--book", default="fanduel", help="Bookmaker key (default: fanduel)")
    parser.add_argument("--bankroll", type=float, help="Optional bankroll for Kelly sizing")
    parser.add_argument(
        "--max-quote-age-minutes", type=float, default=30, help="Max quote age in minutes"
    )
    parser.add_argument(
        "--kickoff-tolerance-seconds", type=float, default=600, help="Kickoff tolerance in seconds"
    )
    args = parser.parse_args()

    try:
        summary = run_compare(
            export_path=args.export,
            season=args.season,
            week=args.week,
            simulations=args.simulations,
            book=args.book,
            bankroll=args.bankroll,
            max_quote_age_minutes=args.max_quote_age_minutes,
            kickoff_tolerance_seconds=args.kickoff_tolerance_seconds,
        )
        print(json.dumps(summary, indent=2, allow_nan=False))
    except Exception as error:
        message = str(error)
        key = os.environ.get("ODDS_API")
        if key:
            message = message.replace(key, "[redacted]")
        parser.exit(1, f"Comparison failed: {message}\n")


if __name__ == "__main__":
    main()
