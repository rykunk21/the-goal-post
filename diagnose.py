"""
diagnose.py — Simulation sanity checker

Run this to validate the model against real game data before trusting edges.

Usage:
    python diagnose.py --gid hou --n-games 5
    python diagnose.py --backtest --gids hou okst drke belm --n-games 5
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent))
import basketball_betting as bb


def diagnose_team(gid: str, n_games: int, xml_cache: Path):
    """
    Load a team's recent games, build their matrix, and compare simulated
    scores against actual final scores recorded in each XML.
    """
    print(f"\n{'='*60}")
    print(f"  Diagnosing: {gid.upper()}  ({n_games} games)")
    print(f"{'='*60}")

    game_ids = bb.get_game_ids_cached(gid, n_games)
    if not game_ids:
        print(f"  No game IDs found for {gid}")
        return

    xmls       = []
    actuals    = []

    for gid_id in game_ids:
        try:
            xml = bb.fetch_game_xml(gid_id, cache_dir=xml_cache)
            xmls.append(xml)
            counts, n_sc, av, ah, vn, hn = bb.xml_to_counts(xml)
            actuals.append({
                "game_id": gid_id,
                "vis":     vn, "home": hn,
                "actual_v": av, "actual_h": ah,
                "n_scoring": n_sc,
            })
            status = f"V {av}-{ah} H" if av is not None else "score N/A"
            print(f"  {gid_id}  {vn} vs {hn}  →  {status}  ({n_sc} transitions)")
        except Exception as e:
            print(f"  {gid_id}  ERROR: {e}")

    if not xmls:
        print("  No XMLs loaded — cannot diagnose")
        return

    # Build this team's matrix using itself as both teams (diagnostic only)
    counts_self, avg_poss, _, _ = bb.blend_team_matrices(xmls, "equal")
    P_self = bb.build_matchup_matrix(counts_self, counts_self)

    # Simulate using avg possessions
    rng = np.random.default_rng(42)
    v_sc, h_sc = bb._run_chain(P_self, n_sim=20000, n_steps=int(avg_poss), rng=rng)

    print(f"\n  Avg possessions (transitions): {avg_poss:.0f}")
    print(f"  Simulated avg score:  V {v_sc.mean():.1f}  H {h_sc.mean():.1f}  "
          f"Total {(v_sc+h_sc).mean():.1f}")

    real_totals  = [a["actual_v"] + a["actual_h"] for a in actuals
                    if a["actual_v"] is not None]
    real_spreads = [a["actual_v"] - a["actual_h"] for a in actuals
                    if a["actual_v"] is not None]
    if real_totals:
        print(f"  Actual avg total:     {sum(real_totals)/len(real_totals):.1f}  "
              f"(games: {real_totals})")
        print(f"  Actual spreads:       {real_spreads}")

    # Print the matrix row sums to check for V/H balance
    row_sums = counts_self.sum(axis=1)
    print(f"\n  Row activity (transitions per state):")
    for i, name in enumerate(bb.STATE_NAMES):
        print(f"    {name:<20}  {row_sums[i]:>8.1f}")

    v_activity = row_sums[:7].sum()
    h_activity = row_sums[7:].sum()
    total_act  = v_activity + h_activity
    print(f"\n  V-states: {v_activity/total_act*100:.1f}%  "
          f"H-states: {h_activity/total_act*100:.1f}%")
    print(f"  (expect ~50/50 for a balanced game)")


def backtest(gids: list[str], n_games: int, xml_cache: Path):
    """
    For each pair combination in gids, simulate the matchup and compare
    to actual results if available.
    """
    print(f"\n{'='*60}")
    print(f"  BACKTEST: {gids}  ({n_games} games each)")
    print(f"{'='*60}")

    team_data = {}
    for gid in gids:
        game_ids = bb.get_game_ids_cached(gid, n_games)
        xmls = []
        for gid_id in game_ids:
            try:
                xmls.append(bb.fetch_game_xml(gid_id, cache_dir=xml_cache))
            except Exception as e:
                print(f"  {gid}/{gid_id} failed: {e}")
        if xmls:
            counts, avg_poss, _, _ = bb.blend_team_matrices(xmls, "recency")
            team_data[gid] = {"counts": counts, "avg_poss": avg_poss}
            print(f"  Loaded {len(xmls)} games for {gid}  (avg {avg_poss:.0f} transitions)")

    print(f"\n  {'Matchup':<20} {'Sim V%':>7} {'SimV':>6} {'SimH':>6} {'Total':>7} {'Spread':>8}")
    print(f"  {'-'*18} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*8}")

    gid_list = list(team_data.keys())
    for i, vis in enumerate(gid_list):
        for j, hom in enumerate(gid_list):
            if i == j:
                continue
            vc = team_data[vis]["counts"]
            hc = team_data[hom]["counts"]
            P  = bb.build_matchup_matrix(vc, hc)
            n_poss = int((team_data[vis]["avg_poss"] + team_data[hom]["avg_poss"]) / 2)
            rng = np.random.default_rng(42)
            v_sc, h_sc = bb._run_chain(P, n_sim=10000, n_steps=n_poss, rng=rng)
            label = f"{vis.upper()} @ {hom.upper()}"
            print(f"  {label:<20} {(v_sc>h_sc).mean()*100:>6.1f}% "
                  f"{v_sc.mean():>6.1f} {h_sc.mean():>6.1f} "
                  f"{(v_sc+h_sc).mean():>7.1f} {(v_sc-h_sc).mean():>+8.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gid",      help="Single team GID to diagnose")
    p.add_argument("--gids",     nargs="+", help="Multiple GIDs for backtest")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--n-games",  type=int, default=5)
    p.add_argument("--xml-dir",  default="game_xml_cache")
    args = p.parse_args()

    xml_cache = Path(args.xml_dir)

    if args.backtest and args.gids:
        backtest(args.gids, args.n_games, xml_cache)
    elif args.gid:
        diagnose_team(args.gid, args.n_games, xml_cache)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
