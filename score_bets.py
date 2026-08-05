#!/usr/bin/env python3
"""
score_bets.py — Score parquet predictions against real ESPN final scores.

Usage:
    python score_bets.py --date 2026-03-07
    python score_bets.py --date 2026-03-07 --bankroll 100 --kelly 0.25 --min-edge 0.05
    python score_bets.py --date 2026-03-07 --parquet path/to/custom.parquet
    python score_bets.py --date 2026-03-07 --no-plot          # table only
    python score_bets.py --score-dir game_xml_cache/sim_results  # score all parquets

Requires:  polars  requests  matplotlib  (pip install polars pyarrow requests matplotlib)
"""
import argparse
import json
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

# ── ESPN scoreboard API ───────────────────────────────────────────────────────
ESPN_API = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/scoreboard"
)


def fetch_espn_results(date: str) -> dict[str, dict]:
    """
    Fetch final scores from ESPN for a given date (YYYY-MM-DD).
    Returns dict keyed by lowercased team name pairs:
        "away_name|||home_name" → {away_score, home_score, away_team, home_team}
    Also indexes by individual team name for fuzzy matching.
    """
    d = date.replace("-", "")
    url = f"{ESPN_API}?dates={d}&groups=50&limit=365"
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read())

    SCOREABLE = {"STATUS_FINAL", "STATUS_IN_PROGRESS", "STATUS_END_PERIOD",
                 "STATUS_HALFTIME"}

    results = {}
    for event in data.get("events", []):
        comp   = event.get("competitions", [{}])[0]
        status = event.get("status", {}).get("type", {}).get("name", "")
        if status not in SCOREABLE:
            continue
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c["homeAway"] == "home"), None)
        away = next((c for c in competitors if c["homeAway"] == "away"), None)
        if not home or not away:
            continue
        try:
            hs = int(home["score"])
            vs = int(away["score"])
        except (KeyError, ValueError):
            continue

        away_name = away["team"]["displayName"].lower()
        home_name = home["team"]["displayName"].lower()
        away_abbr = away["team"]["abbreviation"].lower()
        home_abbr = home["team"]["abbreviation"].lower()

        rec = {
            "away_team":  away["team"]["displayName"],
            "home_team":  home["team"]["displayName"],
            "away_abbr":  away_abbr,
            "home_abbr":  home_abbr,
            "away_score": vs,
            "home_score": hs,
            "total":      vs + hs,
            "spread":     vs - hs,   # positive = away won
            "status":     status,
            "is_final":   status == "STATUS_FINAL",
        }
        # Index by multiple keys for robust matching
        results[f"{away_name}|||{home_name}"] = rec
        results[f"{away_abbr}|||{home_abbr}"] = rec
        results[away_name] = rec
        results[home_name] = rec
        results[away_abbr] = rec
        results[home_abbr] = rec

    return results


def _find_result(row: dict, results: dict) -> dict | None:
    """Try multiple keys to match a parquet row to an ESPN result."""
    away = (row.get("away_team") or "").lower()
    home = (row.get("home_team") or "").lower()

    # Parse abbreviated game label e.g. "HOU @ OKST"
    game  = row.get("game", "")
    parts = game.split(" @ ")
    abbr_away = parts[0].strip().lower() if len(parts) == 2 else ""
    abbr_home = parts[1].strip().lower() if len(parts) == 2 else ""

    for key in [
        f"{away}|||{home}",
        f"{abbr_away}|||{abbr_home}",
        away, home, abbr_away, abbr_home,
    ]:
        if key and key in results:
            return results[key]
    return None


# ── Bet outcome evaluation ─────────────────────────────────────────────────────

def evaluate_bet(row: dict, result: dict) -> str:
    """
    Return "win", "loss", or "push" for a single bet row against a real result.
    """
    market = row["market"]
    side   = row["side"].upper()
    away_s = result["away_score"]
    home_s = result["home_score"]

    if market == "ML":
        # Side is like "XAVI ML" or "NOVA ML"
        # We check who won against the moneyline side token
        side_tok = side.split()[0]
        if side_tok == result["away_abbr"].upper() or \
           side_tok in result["away_team"].upper().split():
            return "win" if away_s > home_s else "loss"
        else:
            return "win" if home_s > away_s else "loss"

    if market == "Spread":
        # Side is like "PVAM +35.0" or "KTY -3.5"
        tokens = side.split()
        if len(tokens) < 2:
            return "?"
        try:
            line = float(tokens[-1])
        except ValueError:
            return "?"

        side_tok = tokens[0]  # e.g. "PVAM", "KTY", "XAVI"

        # Use the game field "AWAY_GID @ HOME_GID" as ground truth for away/home
        # This is more reliable than fuzzy team name matching
        game_parts = row.get("game", "").upper().split(" @ ")
        game_away_tok = game_parts[0].strip() if len(game_parts) == 2 else ""
        game_home_tok = game_parts[1].strip() if len(game_parts) == 2 else ""

        if side_tok == game_away_tok:
            is_away = True
        elif side_tok == game_home_tok:
            is_away = False
        else:
            # Fallback: check ESPN abbrs
            is_away = (side_tok == result["away_abbr"].upper())

        # ATS result: team_score + spread vs opponent
        if is_away:
            covered = (away_s + line) > home_s
            push    = (away_s + line) == home_s
        else:
            covered = (home_s + line) > away_s
            push    = (home_s + line) == away_s
        if push:
            return "push"
        return "win" if covered else "loss"

    if market == "Total":
        tokens = side.split()
        # Side like "OVER 152.5" or "UNDER 148.5"
        direction = tokens[0].upper() if tokens else ""
        try:
            line = float(tokens[-1])
        except ValueError:
            return "?"
        total = result["total"]
        if total == line:
            return "push"
        if direction == "OVER":
            return "win" if total > line else "loss"
        else:
            return "win" if total < line else "loss"

    return "?"


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal (profit per $1 wagered)."""
    if odds >= 100:
        return odds / 100
    else:
        return 100 / abs(odds)


# ── Scoring engine ─────────────────────────────────────────────────────────────

def score_parquet(
    parquet_path: Path,
    date: str,
    bankroll: float = 100.0,
    kelly_frac: float = 0.25,
    min_edge: float = 0.0,
    markets: list[str] | None = None,
) -> pl.DataFrame:
    """
    Load a parquet, fetch ESPN results, score each bet, return enriched DataFrame.
    """
    df = pl.read_parquet(parquet_path)

    # Apply filters
    if min_edge > 0:
        df = df.filter(pl.col("edge") > min_edge)
    if markets:
        market_map = {"spread": "Spread", "ml": "ML", "total": "Total"}
        allowed = [market_map[m] for m in markets if m in market_map]
        df = df.filter(pl.col("market").is_in(allowed))
    df = df.sort("edge", descending=True)

    print(f"  Fetching ESPN results for {date} …")
    espn = fetch_espn_results(date)
    print(f"  Found {len([k for k in espn if '|||' in k])} completed games on ESPN")

    rows = df.to_dicts()
    enriched = []
    skipped_pending = 0
    for row in rows:
        result = _find_result(row, espn)
        if result is None:
            # Game hasn't started or isn't on ESPN — skip entirely
            skipped_pending += 1
            continue
        outcome     = evaluate_bet(row, result)
        away_score  = result["away_score"]
        home_score  = result["home_score"]
        is_final    = result["is_final"]
        game_status = result["status"]
        # For in-progress games, mark outcome as provisional
        if not is_final and outcome in ("win", "loss"):
            outcome = outcome + "_live"

        if row["market"] == "Spread":
            actual_line = float(result["spread"])
        elif row["market"] == "Total":
            actual_line = float(result["total"])
        else:
            actual_line = None

        row["outcome"]     = outcome
        row["away_score"]  = away_score
        row["home_score"]  = home_score
        row["actual_line"] = actual_line
        row["is_final"]    = is_final
        row["game_status"] = game_status
        enriched.append(row)

    if skipped_pending:
        print(f"  Skipped {skipped_pending} bets — games not yet started or no ESPN data")
    print(f"  Scoring {len(enriched)} bets across "
          f"{sum(1 for r in enriched if r['is_final'])} final / "
          f"{sum(1 for r in enriched if not r['is_final'])} in-progress games")

    # Sequential Kelly P&L — only count final games
    bal = bankroll
    for row in enriched:
        bet_size = bal * row["kelly"]
        row["bankroll_before"] = bal
        row["bet_size"]        = bet_size
        outcome = row["outcome"]
        if outcome == "win":
            payout       = bet_size * american_to_decimal(row["odds"])
            row["pnl"]   = payout
            bal         += payout
        elif outcome == "loss":
            row["pnl"]   = -bet_size
            bal         -= bet_size
        elif outcome == "push":
            row["pnl"]   = 0.0
        else:
            # live/pending — don't move the bankroll, mark P&L null
            row["pnl"]   = None
        row["bankroll_after"] = bal

    schema_overrides = {
        "away_score":      pl.Int32,
        "home_score":      pl.Int32,
        "actual_line":     pl.Float64,
        "pnl":             pl.Float64,
        "bet_size":        pl.Float64,
        "bankroll_before": pl.Float64,
        "bankroll_after":  pl.Float64,
        "is_final":        pl.Boolean,
        "game_status":     pl.Utf8,
    }
    return pl.DataFrame(enriched).with_columns([
        pl.col(c).cast(t) for c, t in schema_overrides.items()
        if c in pl.DataFrame(enriched).columns
    ])


# ── Console report ─────────────────────────────────────────────────────────────

def print_report(scored: pl.DataFrame, date: str, bankroll: float) -> None:
    graded = scored.filter(pl.col("outcome").is_in(["win", "loss", "push"]))
    wins   = (graded["outcome"] == "win").sum()
    losses = (graded["outcome"] == "loss").sum()
    pushes = (graded["outcome"] == "push").sum()
    total  = wins + losses
    pct    = wins / total * 100 if total else 0

    pnl_rows  = scored.filter(pl.col("pnl").is_not_null())
    total_pnl = pnl_rows["pnl"].sum()
    final_bal = bankroll + total_pnl

    SEP = "═" * 110
    print(f"\n{SEP}")
    print(f"  SCORECARD  —  {date}")
    print(f"  {wins}W / {losses}L / {pushes}P  ({pct:.1f}% ATS)  |  "
          f"P&L: ${total_pnl:+.2f}  |  "
          f"Bankroll: ${bankroll:.2f} → ${final_bal:.2f}  "
          f"({'▲' if total_pnl >= 0 else '▼'} {abs(total_pnl/bankroll*100):.1f}%)")
    print(SEP)
    print(f"  {'#':<4} {'Away':<22} {'Home':<20} {'Mkt':<8} {'Side':<22} "
          f"{'Edge':>7} {'Bet $':>7} {'Score':>9} {'Result':<7} {'P&L':>8}")
    print(f"  {'─'*3} {'─'*21} {'─'*19} {'─'*7} {'─'*21} "
          f"{'─'*7} {'─'*7} {'─'*9} {'─'*6} {'─'*8}")

    ICONS = {"win": "✅", "loss": "❌", "push": "➖",
             "win_live": "🟢", "loss_live": "🔴", "no_data": "❓", "?": "❓"}

    # Split final vs live
    final_rows = [r for r in scored.iter_rows(named=True)
                  if r.get("is_final", True) and r["outcome"] in ("win","loss","push")]
    live_rows  = [r for r in scored.iter_rows(named=True)
                  if not r.get("is_final", True)]

    def _print_rows(rows, start_i=1):
        for i, row in enumerate(rows, start_i):
            away  = (row.get("away_team") or "")[:21]
            home  = (row.get("home_team") or "")[:19]
            score = (f"{row['away_score']}-{row['home_score']}"
                     if row.get("away_score") is not None else "—")
            icon  = ICONS.get(row["outcome"], "❓")
            pnl   = f"${row['pnl']:+.2f}" if row.get("pnl") is not None else "—"
            bet   = f"${row['bet_size']:.2f}" if row.get("bet_size") else "—"
            print(f"  {i:<4} {away:<22} {home:<20} {row['market']:<8} "
                  f"{row['side'][:21]:<22} {row['edge']*100:>+6.1f}% "
                  f"{bet:>7} {score:>9} {icon} {row['outcome']:<10} {pnl:>8}")

    print(f"\n{SEP}")
    print(f"  SCORECARD  —  {date}")
    print(f"  {wins}W / {losses}L / {pushes}P  ({pct:.1f}% ATS)  |  "
          f"P&L: ${total_pnl:+.2f}  |  "
          f"Bankroll: ${bankroll:.2f} → ${final_bal:.2f}  "
          f"({'▲' if total_pnl >= 0 else '▼'} {abs(total_pnl/bankroll*100):.1f}%)"
          + (f"  |  🔴 {len(live_rows)} live/pending" if live_rows else ""))
    print(SEP)
    print(f"  {'#':<4} {'Away':<22} {'Home':<20} {'Mkt':<8} {'Side':<22} "
          f"{'Edge':>7} {'Bet $':>7} {'Score':>9} {'Result':<11} {'P&L':>8}")
    print(f"  {'─'*3} {'─'*21} {'─'*19} {'─'*7} {'─'*21} "
          f"{'─'*7} {'─'*7} {'─'*9} {'─'*10} {'─'*8}")

    _print_rows(final_rows, 1)

    if live_rows:
        print(f"\n  ── IN PROGRESS / LIVE (not yet counted in P&L) ──")
        _print_rows(live_rows, len(final_rows) + 1)

    print(SEP)

    # Market breakdown
    for mkt in ["Spread", "ML", "Total"]:
        sub = graded.filter(pl.col("market") == mkt)
        if len(sub) == 0:
            continue
        w = (sub["outcome"] == "win").sum()
        l = (sub["outcome"] == "loss").sum()
        p_pct = w / (w + l) * 100 if (w + l) else 0
        sub_pnl = scored.filter(pl.col("market") == mkt)["pnl"].drop_nulls().sum()
        print(f"  {mkt:<10}  {w}W/{l}L  {p_pct:.0f}%  P&L ${sub_pnl:+.2f}")
    print()


# ── Visualization ──────────────────────────────────────────────────────────────

PALETTE = {
    "bg":       "#0d0f14",
    "panel":    "#13161e",
    "border":   "#1e2230",
    "win":      "#00e5a0",
    "loss":     "#ff4d6d",
    "push":     "#f5c518",
    "neutral":  "#4a5068",
    "text":     "#e8eaf0",
    "subtext":  "#6b7394",
    "accent":   "#4f8ef7",
}


def make_visualization(scored: pl.DataFrame, date: str,
                        bankroll: float, output_path: Path) -> None:
    graded  = scored.filter(pl.col("outcome").is_in(["win", "loss", "push"]))
    wins    = (graded["outcome"] == "win").sum()
    losses  = (graded["outcome"] == "loss").sum()
    pushes  = (graded["outcome"] == "push").sum()
    total   = wins + losses
    win_pct = wins / total * 100 if total else 0

    pnl_col    = scored.filter(pl.col("pnl").is_not_null())["pnl"]
    total_pnl  = pnl_col.sum()
    final_bal  = bankroll + total_pnl
    roi        = total_pnl / bankroll * 100

    # Cumulative bankroll series
    bal_series = [bankroll]
    for row in scored.filter(pl.col("pnl").is_not_null()).iter_rows(named=True):
        bal_series.append(bal_series[-1] + row["pnl"])

    # Edge vs outcome scatter data
    scatter_rows = scored.filter(pl.col("pnl").is_not_null()).to_dicts()

    fig = plt.figure(figsize=(18, 11), facecolor=PALETTE["bg"])
    fig.patch.set_facecolor(PALETTE["bg"])

    gs = GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38,
                  left=0.05, right=0.97, top=0.90, bottom=0.07)

    # ── Helper ────────────────────────────────────────────────────────────────
    def styled_ax(ax, title=""):
        ax.set_facecolor(PALETTE["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
            spine.set_linewidth(0.8)
        ax.tick_params(colors=PALETTE["subtext"], labelsize=8)
        ax.xaxis.label.set_color(PALETTE["subtext"])
        ax.yaxis.label.set_color(PALETTE["subtext"])
        if title:
            ax.set_title(title, color=PALETTE["text"], fontsize=9,
                         fontweight="bold", pad=8, loc="left")
        return ax

    # ── 1. Header stats ───────────────────────────────────────────────────────
    ax_hdr = fig.add_axes([0.0, 0.92, 1.0, 0.08])
    ax_hdr.set_facecolor(PALETTE["bg"])
    ax_hdr.axis("off")

    pnl_color = PALETTE["win"] if total_pnl >= 0 else PALETTE["loss"]
    ax_hdr.text(0.02, 0.5, f"SCORECARD  ·  {date}",
                transform=ax_hdr.transAxes, color=PALETTE["text"],
                fontsize=15, fontweight="bold", va="center",
                fontfamily="monospace")
    stats = [
        (f"{wins}W / {losses}L / {pushes}P", PALETTE["text"]),
        (f"{win_pct:.1f}% ATS", PALETTE["accent"]),
        (f"P&L  ${total_pnl:+.2f}", pnl_color),
        (f"ROI  {roi:+.1f}%", pnl_color),
        (f"${bankroll:.0f} → ${final_bal:.2f}", PALETTE["subtext"]),
    ]
    x = 0.36
    for txt, col in stats:
        ax_hdr.text(x, 0.5, txt, transform=ax_hdr.transAxes,
                    color=col, fontsize=11, va="center",
                    fontfamily="monospace", fontweight="bold")
        x += 0.13

    # ── 2. Cumulative bankroll (spans top 2 rows, left 2 cols) ─────────────
    ax_bal = fig.add_subplot(gs[:2, :2])
    styled_ax(ax_bal, "Cumulative Bankroll")

    xs = list(range(len(bal_series)))
    # shade region
    ax_bal.fill_between(xs, bankroll, bal_series,
                         where=[b >= bankroll for b in bal_series],
                         alpha=0.15, color=PALETTE["win"], interpolate=True)
    ax_bal.fill_between(xs, bankroll, bal_series,
                         where=[b < bankroll for b in bal_series],
                         alpha=0.15, color=PALETTE["loss"], interpolate=True)
    ax_bal.plot(xs, bal_series, color=PALETTE["accent"], linewidth=2, zorder=3)
    ax_bal.axhline(bankroll, color=PALETTE["border"], linewidth=1,
                   linestyle="--", alpha=0.7)

    # Annotate final value
    final_c = PALETTE["win"] if bal_series[-1] >= bankroll else PALETTE["loss"]
    ax_bal.annotate(f"${bal_series[-1]:.2f}",
                    xy=(xs[-1], bal_series[-1]),
                    xytext=(-30, 10), textcoords="offset points",
                    color=final_c, fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=final_c, lw=0.8))

    # Color dots by outcome
    outcome_list = scored.filter(pl.col("pnl").is_not_null())["outcome"].to_list()
    dot_colors   = [PALETTE["win"] if o == "win"
                    else PALETTE["loss"] if o == "loss"
                    else PALETTE["push"] for o in outcome_list]
    ax_bal.scatter(xs[1:], bal_series[1:], c=dot_colors, s=30, zorder=4)
    ax_bal.set_ylabel("Balance ($)")
    ax_bal.set_xlabel("Bet # (sorted by edge)")

    # ── 3. Win/loss donut ─────────────────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[0, 2])
    styled_ax(ax_pie, "Outcomes")
    vals   = [wins, losses, pushes]
    cols   = [PALETTE["win"], PALETTE["loss"], PALETTE["push"]]
    labels = [f"W {wins}", f"L {losses}", f"P {pushes}"]
    non_z  = [(v, c, l) for v, c, l in zip(vals, cols, labels) if v > 0]
    if non_z:
        vv, cc, ll = zip(*non_z)
        wedges, _ = ax_pie.pie(
            vv, colors=cc, startangle=90,
            wedgeprops=dict(width=0.5, edgecolor=PALETTE["bg"], linewidth=2)
        )
        ax_pie.text(0, 0, f"{win_pct:.0f}%\nATS", ha="center", va="center",
                    color=PALETTE["text"], fontsize=11, fontweight="bold")
        ax_pie.legend(wedges, ll, loc="lower center", fontsize=7,
                      labelcolor=PALETTE["subtext"],
                      facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
                      bbox_to_anchor=(0.5, -0.12), ncol=3)

    # ── 4. P&L by market bar chart ─────────────────────────────────────────
    ax_mkt = fig.add_subplot(gs[0, 3])
    styled_ax(ax_mkt, "P&L by Market")
    markets = ["Spread", "ML", "Total"]
    pnls    = []
    for m in markets:
        sub = scored.filter(pl.col("market") == m)["pnl"].drop_nulls()
        pnls.append(sub.sum() if len(sub) else 0.0)
    bar_cols = [PALETTE["win"] if p >= 0 else PALETTE["loss"] for p in pnls]
    bars = ax_mkt.bar(markets, pnls, color=bar_cols,
                      edgecolor=PALETTE["bg"], linewidth=0.5)
    ax_mkt.axhline(0, color=PALETTE["border"], linewidth=0.8)
    for bar, val in zip(bars, pnls):
        ax_mkt.text(bar.get_x() + bar.get_width() / 2,
                    val + (0.3 if val >= 0 else -0.6),
                    f"${val:+.2f}", ha="center", va="bottom",
                    color=PALETTE["text"], fontsize=8)
    ax_mkt.set_ylabel("P&L ($)")
    ax_mkt.tick_params(axis="x", labelsize=8)

    # ── 5. Edge vs P&L scatter ─────────────────────────────────────────────
    ax_scat = fig.add_subplot(gs[1, 2:])
    styled_ax(ax_scat, "Model Edge vs P&L")
    for row in scatter_rows:
        c = (PALETTE["win"] if row["outcome"] == "win"
             else PALETTE["loss"] if row["outcome"] == "loss"
             else PALETTE["push"])
        ax_scat.scatter(row["edge"] * 100, row["pnl"],
                        color=c, s=50, alpha=0.85, zorder=3,
                        edgecolors=PALETTE["bg"], linewidth=0.5)
        # label the dot with the side abbreviation
        lbl = row["side"].split()[0]
        ax_scat.annotate(lbl, (row["edge"] * 100, row["pnl"]),
                         fontsize=6, color=PALETTE["subtext"],
                         xytext=(3, 3), textcoords="offset points")
    ax_scat.axhline(0, color=PALETTE["border"], linewidth=0.8, linestyle="--")
    ax_scat.axvline(0, color=PALETTE["border"], linewidth=0.8, linestyle="--")
    ax_scat.set_xlabel("Model Edge (%)")
    ax_scat.set_ylabel("P&L ($)")
    win_p  = mpatches.Patch(color=PALETTE["win"],  label="Win")
    loss_p = mpatches.Patch(color=PALETTE["loss"], label="Loss")
    push_p = mpatches.Patch(color=PALETTE["push"], label="Push")
    ax_scat.legend(handles=[win_p, loss_p, push_p], fontsize=7,
                   facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
                   labelcolor=PALETTE["text"])

    # ── 6. Per-bet P&L waterfall ─────────────────────────────────────────────
    ax_wf = fig.add_subplot(gs[2, :])
    styled_ax(ax_wf, "Per-Bet P&L  (sorted by edge)")
    pnl_vals  = [r["pnl"]     for r in scatter_rows]
    sides     = [r["side"].split()[0] for r in scatter_rows]
    outcomes  = [r["outcome"] for r in scatter_rows]
    bar_c     = [PALETTE["win"] if o == "win"
                 else PALETTE["loss"] if o == "loss"
                 else PALETTE["push"] for o in outcomes]
    xpos      = list(range(len(pnl_vals)))
    ax_wf.bar(xpos, pnl_vals, color=bar_c,
              edgecolor=PALETTE["bg"], linewidth=0.4, width=0.7)
    ax_wf.axhline(0, color=PALETTE["border"], linewidth=0.8)
    ax_wf.set_xticks(xpos)
    ax_wf.set_xticklabels(sides, rotation=45, ha="right", fontsize=7,
                           color=PALETTE["subtext"])
    ax_wf.set_ylabel("P&L ($)")

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Chart saved → {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_args():
    p = argparse.ArgumentParser(description="Score bet predictions vs real results")
    p.add_argument("--date",       default=None,
                   help="Date to score (YYYY-MM-DD). Defaults to yesterday.")
    p.add_argument("--parquet",    default=None,
                   help="Explicit parquet path (auto-detected if omitted)")
    p.add_argument("--score-dir",  default="game_xml_cache/sim_results",
                   help="Score all parquets in this directory")
    p.add_argument("--all",        action="store_true",
                   help="Score every parquet in --score-dir")
    p.add_argument("--bankroll",   type=float, default=100.0)
    p.add_argument("--kelly",      type=float, default=0.25,
                   help="Kelly fraction used in original screener run")
    p.add_argument("--min-edge",   type=float, default=0.0,
                   help="Filter: only score bets above this edge (e.g. 0.05)")
    p.add_argument("--markets",    nargs="+", default=None,
                   choices=["spread", "ml", "total"],
                   help="Filter to specific markets, e.g. --markets spread ml")
    p.add_argument("--no-plot",    action="store_true",
                   help="(deprecated — charts are opt-in via --plot)")
    p.add_argument("--plot",       action="store_true",
                   help="Save a scorecard PNG chart alongside the parquet")
    p.add_argument("--output",     default=None,
                   help="Chart output path (default: scorecard_YYYY-MM-DD.png)")
    return p


def score_one(date: str, parquet_path: Path, args) -> None:
    print(f"\n  Scoring: {parquet_path.name}  (date={date})")
    try:
        scored = score_parquet(parquet_path, date,
                               bankroll=args.bankroll,
                               kelly_frac=args.kelly,
                               min_edge=args.min_edge,
                               markets=getattr(args, "markets", None))
    except Exception as e:
        print(f"  ERROR during scoring: {e}")
        return

    print_report(scored, date, args.bankroll)

    if getattr(args, "plot", False):
        out = Path(args.output) if args.output else \
              parquet_path.parent / f"scorecard_{date}.png"
        try:
            make_visualization(scored, date, args.bankroll, out)
        except Exception as e:
            print(f"  WARNING: chart failed: {e}")

    # Optionally save enriched parquet alongside original
    enriched_path = parquet_path.with_name(parquet_path.stem + "_scored.parquet")
    scored.write_parquet(enriched_path)
    print(f"  Enriched parquet → {enriched_path}")


def main():
    args = build_args().parse_args()

    sim_dir = Path(args.score_dir)

    if args.all:
        parquets = sorted(sim_dir.glob("????-??-??.parquet"))
        if not parquets:
            sys.exit(f"No parquets found in {sim_dir}")
        for pq in parquets:
            date = pq.stem
            score_one(date, pq, args)
        return

    # Single date
    if args.date:
        date = args.date
    else:
        date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    if args.parquet:
        pq = Path(args.parquet)
    else:
        pq = sim_dir / f"{date}.parquet"

    if not pq.exists():
        sys.exit(f"Parquet not found: {pq}\n"
                 f"Run the screener first:  python basketball_betting.py --screen ...")

    score_one(date, pq, args)


if __name__ == "__main__":
    main()
