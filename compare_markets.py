#!/usr/bin/env python3
"""Parse markets.html, load fresh simulation PMFs, and compare spreads/totals.

Uses FanDuel MA column for market lines (avoids Polymarket/Kalshi outliers).
"""

import csv
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Team name mapping: full name -> abbreviation
# ---------------------------------------------------------------------------

TEAM_NAME_MAP = {
    "Cardinals": "ARI",
    "Falcons": "ATL",
    "Ravens": "BAL",
    "Bills": "BUF",
    "Panthers": "CAR",
    "Bears": "CHI",
    "Bengals": "CIN",
    "Browns": "CLE",
    "Cowboys": "DAL",
    "Broncos": "DEN",
    "Lions": "DET",
    "Packers": "GB",
    "Texans": "HOU",
    "Colts": "IND",
    "Jaguars": "JAX",
    "Chiefs": "KC",
    "Rams": "LA",
    "Chargers": "LAC",
    "Raiders": "LV",
    "Dolphins": "MIA",
    "Vikings": "MIN",
    "Patriots": "NE",
    "Saints": "NO",
    "Giants": "NYG",
    "Jets": "NYJ",
    "Eagles": "PHI",
    "Steelers": "PIT",
    "Seahawks": "SEA",
    "49ers": "SF",
    "Buccaneers": "TB",
    "Titans": "TEN",
    "Commanders": "WAS",
}


def full_to_abbr(name: str) -> str | None:
    """Convert full team name (e.g. 'Jaguars') to abbreviation (e.g. 'JAX')."""
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    parts = name.strip().split()
    if parts:
        last = parts[-1]
        if last in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[last]
    return None


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

def parse_markets_html(html_path: Path) -> list[dict]:
    """Extract game matchups and FanDuel spreads from Action Network HTML.

    Column mapping (after game-info TD):
      0 = Open
      1 = Best Odds (may have Polymarket/Kalshi)
      2 = N/A
      3 = BetMGM MA
      4 = Caesars MA
      5 = DK MA
      6 = FanDuel MA  <-- use this
      7 = Fanatics MA
      8 = Polymarket
      9 = Kalshi
      10 = BetMGM NJ
      11 = Fanatics NJ
    """
    raw = html_path.read_text(encoding="utf-8")

    games = []
    all_trs = re.findall(r'<tr>(.*?)</tr>', raw, re.DOTALL)

    i = 0
    while i < len(all_trs):
        tr = all_trs[i]
        if 'best-odds__game-info' not in tr:
            i += 1
            continue

        team_names = re.findall(r'alt="([^"]+) Team Icon"', tr)
        if len(team_names) != 2:
            names = re.findall(r'<div class="game-info__team--desktop"><span>([^<]+)</span>', tr)
            if len(names) >= 2:
                away = names[0].strip()
                home = names[1].strip()
            else:
                i += 1
                continue
        else:
            away = team_names[0]
            home = team_names[1]

        tds = re.findall(r'<td[^\u003e]*>(.*?)</td>', tr)
        if len(tds) < 7:
            i += 1
            continue

        # Extract only spread values (css-1jlt5rt class), skipping juice
        fanduel_nums = re.findall(r'css-1jlt5rt[^"\u003e]*">([+-]?\d+(?:\.5)?)</span>', tds[6])
        if len(fanduel_nums) < 2:
            fanduel_nums = re.findall(r'css-1jlt5rt[^"\u003e]*">([+-]?\d+(?:\.5)?)</span>', tds[5])
            if len(fanduel_nums) < 2:
                fanduel_nums = re.findall(r'css-1jlt5rt[^"\u003e]*">([+-]?\d+(?:\.5)?)</span>', tds[4])

        away_spread = fanduel_nums[0] if len(fanduel_nums) >= 1 else None
        home_spread = fanduel_nums[1] if len(fanduel_nums) >= 2 else None

        # Look for kickoff in next row
        kickoff = None
        if i + 1 < len(all_trs):
            next_tr = all_trs[i + 1]
            if 'best-odds__game-status' in next_tr:
                kickoff_match = re.search(r'<div[^\u003e]*>([^\u003c]*(?:AM|PM)[^\u003c]*)', next_tr)
                if kickoff_match:
                    kickoff = kickoff_match.group(1).strip()

        games.append({
            "home": home.strip(),
            "away": away.strip(),
            "away_spread": away_spread,
            "home_spread": home_spread,
            "kickoff": kickoff,
        })
        i += 1

    return games


# ---------------------------------------------------------------------------
# Simulation PMF helpers
# ---------------------------------------------------------------------------

def load_pmf(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def spread_probabilities(home_margin_pmf: pd.DataFrame, line: float) -> dict:
    """
    For a spread line on the home team,
    compute P(home covers), P(push), P(home fails to cover).

    Home covers if home_margin > -line.
    e.g. home +10 covers if margin > -10.
    """
    win = home_margin_pmf[home_margin_pmf["points"] > -line]["probability"].sum()
    push = home_margin_pmf[home_margin_pmf["points"] == -line]["probability"].sum()
    loss = home_margin_pmf[home_margin_pmf["points"] < -line]["probability"].sum()
    return {"home_win": win, "push": push, "home_loss": loss}


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

# Use the latest weekly export
EXPORT_DIR = Path("artifacts/transitions/weekly/20261105T235437089506Z")
MARKETS_HTML = Path("markets.html")


def main():
    # 1. Parse markets HTML
    games = parse_markets_html(MARKETS_HTML)
    print(f"Parsed {len(games)} games from markets.html")

    # 2. Load schedule to map display names to game IDs
    schedule = pd.read_csv(EXPORT_DIR / "schedule.csv")

    # Build game_id -> (home, away) from schedule
    game_id_teams = {}
    for _, row in schedule.iterrows():
        game_id_teams[row["game_id"]] = (row["home_team"], row["away_team"])

    results = []
    for game in games:
        home = game["home"]
        away = game["away"]
        home_spread_raw = game.get("home_spread")

        if home_spread_raw is None:
            continue

        # Convert full names to abbreviations
        home_abbr = full_to_abbr(home)
        away_abbr = full_to_abbr(away)

        if home_abbr is None or away_abbr is None:
            print(f"  WARNING: Could not map team names for {away} @ {home}")
            continue

        # Find matching game_id
        game_id = None
        for gid, (h, a) in game_id_teams.items():
            if (h == home_abbr and a == away_abbr) or (h == away_abbr and a == home_abbr):
                game_id = gid
                # Ensure orientation matches markets.html
                if h != home_abbr:
                    home_abbr, away_abbr = away_abbr, home_abbr
                break

        if game_id is None:
            print(f"  WARNING: No simulation found for {away} @ {home} ({away_abbr} @ {home_abbr})")
            continue

        total_pmf_path = EXPORT_DIR / game_id / "total-pmf.csv"
        margin_pmf_path = EXPORT_DIR / game_id / "home_minus_away-pmf.csv"

        if not total_pmf_path.exists() or not margin_pmf_path.exists():
            print(f"  WARNING: Missing PMFs for {game_id}")
            continue

        total_pmf = load_pmf(total_pmf_path)
        margin_pmf = load_pmf(margin_pmf_path)

        # Parse home spread line from the market
        try:
            home_spread = float(home_spread_raw)
        except (ValueError, TypeError):
            continue

        spread_probs = spread_probabilities(margin_pmf, home_spread)

        # For totals: compute fair total (median)
        cum_prob = total_pmf["probability"].cumsum()
        median_idx = (cum_prob - 0.5).abs().argmin()
        median_total = total_pmf["points"].iloc[median_idx]

        # Compute fair spread (median of margin distribution)
        cum_margin = margin_pmf["probability"].cumsum()
        median_m_idx = (cum_margin - 0.5).abs().argmin()
        median_margin = margin_pmf["points"].iloc[median_m_idx]

        # Compute expected total from PMF
        expected_total = (total_pmf["points"] * total_pmf["probability"]).sum()

        # Simulation probabilities for the market line
        p_home_cover = spread_probs["home_win"]
        p_away_cover = spread_probs["home_loss"]
        p_push = spread_probs["push"]

        # Edge: how much more likely is home to cover vs away?
        spread_edge_home = (p_home_cover - p_away_cover) * 100

        results.append({
            "game_id": game_id,
            "away": away_abbr,
            "home": home_abbr,
            "kickoff": game.get("kickoff", ""),
            "market_spread_home": home_spread,
            "sim_median_margin": median_margin,
            "sim_median_total": median_total,
            "sim_expected_total": expected_total,
            "sim_p_home_cover": p_home_cover,
            "sim_p_away_cover": p_away_cover,
            "sim_p_push": p_push,
            "spread_edge_home_pct": spread_edge_home,
        })

    if not results:
        print("No matching games found between markets and simulations.")
        return

    df = pd.DataFrame(results)

    # Rank by absolute spread edge (simulation vs market disagreement)
    df["abs_spread_edge"] = df["spread_edge_home_pct"].abs()
    df = df.sort_values("abs_spread_edge", ascending=False)

    # Print summary table
    print("\n" + "=" * 130)
    print("NFL SIMULATION vs MARKET SPREAD COMPARISON")
    print("=" * 130)
    print(
        f"{'Rank':<6}{'Game':<25}{'Market Spread':<18}{'Sim Median':<14}{'P(Home Cover)':<16}{'P(Away Cover)':<16}{'Edge (Home %)':<14}{'Sim Total':<12}"
    )
    print("-" * 130)
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        game_str = f"{row['away']} @ {row['home']}"
        market = f"{row['market_spread_home']:+.1f} ({row['home']})"
        sim_med = f"{row['sim_median_margin']:+.1f}"
        p_home = f"{row['sim_p_home_cover']:.2%}"
        p_away = f"{row['sim_p_away_cover']:.2%}"
        edge = f"{row['spread_edge_home_pct']:+.1f}%"
        total = f"{row['sim_median_total']:.1f}"
        print(f"{rank:<6}{game_str:<25}{market:<18}{sim_med:<14}{p_home:<16}{p_away:<16}{edge:<14}{total:<12}")

    print("\n" + "=" * 130)
    print("INTERPRETATION:")
    print("  - Positive Edge(Home): Simulation favors home team more than market")
    print("  - Negative Edge(Home): Simulation favors away team more than market")
    print("  - Higher absolute edge = larger disagreement between model and market")
    print("=" * 130)

    # Save to CSV
    output_path = Path("simulation_vs_market_comparison.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
