#!/usr/bin/env python3
"""
Basketball Betting Simulator
=============================
Scrapes the last N games for two teams from StatBroadcast, builds
blended MCMC transition matrices, simulates the upcoming matchup,
and queries the resulting distributions against betting lines.

Usage examples
--------------
# Full pipeline: scrape + simulate + query lines
python basketball_betting.py \\
    --visitor duke --home unc \\
    --total 155.5 --spread -4.5 \\
    --visitor-ml -180 --home-ml +155

# Use cached XMLs (skip scraping)
python basketball_betting.py \\
    --visitor duke --home unc \\
    --xml-dir ./game_xmls \\
    --total 155.5 --spread -4.5

# Just simulate, no betting lines
python basketball_betting.py --visitor duke --home unc

Odds formats
------------
American odds:  -110, +150, -180, +220  (default)
Decimal odds:   use --odds-format decimal
"""

import argparse
import json
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── MCMC state space (14 states, identical to basketball_mcmc.py) ─────────────
STATE_NAMES = [
    "V_made_2", "V_made_3", "V_made_ft",
    "V_miss_2", "V_miss_3", "V_miss_ft", "V_turnover",
    "H_made_2", "H_made_3", "H_made_ft",
    "H_miss_2", "H_miss_3", "H_miss_ft", "H_turnover",
]
N_STATES  = len(STATE_NAMES)
STATE_IDX = {s: i for i, s in enumerate(STATE_NAMES)}

POINTS_ARR = np.array([
    [2,0],[3,0],[1,0],[0,0],[0,0],[0,0],[0,0],
    [0,2],[0,3],[0,1],[0,0],[0,0],[0,0],[0,0],
], dtype=np.int32)

V_PTS = POINTS_ARR[:, 0]
H_PTS = POINTS_ARR[:, 1]

# ── StatBroadcast URLs ────────────────────────────────────────────────────────
SB_BASE       = "https://www.statbroadcast.com"
SB_STATMONITR = f"{SB_BASE}/events/statmonitr.php"
SB_ARCHIVE    = f"{SB_BASE}/events/_archive.php"
SB_XML        = f"{SB_BASE}/xml/xml.php"
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
GID_CACHE_FILE        = Path("game_xml_cache/team_gids.json")
NAME_TO_GID_CACHE_FILE = Path("game_xml_cache/name_to_gid.json")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SCRAPING
# ═══════════════════════════════════════════════════════════════════════════════

SB_EVENTS      = f"{SB_BASE}/events/"
ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/scoreboard"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  ODDS API + ESPN GAME DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_espn_games(date: str) -> list[dict]:
    """
    Fetch all NCAAB games for a given date from ESPN's public scoreboard API.
    No auth required. Returns every D-I game, not just ones books offer lines on.

    date: 'YYYY-MM-DD'
    """
    date_compact = date.replace("-", "")
    params = {"dates": date_compact, "groups": 50, "limit": 365}
    resp = requests.get(ESPN_SCOREBOARD, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for event in data.get("events", []):
        comps = event.get("competitions", [{}])[0]
        competitors = comps.get("competitors", [])
        if len(competitors) < 2:
            continue
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        games.append({
            "espn_id":   event.get("id"),
            "name":      event.get("name", ""),
            "away_team": away["team"]["displayName"],
            "away_abbr": away["team"]["abbreviation"],
            "home_team": home["team"]["displayName"],
            "home_abbr": home["team"]["abbreviation"],
            "status":    event.get("status", {}).get("type", {}).get("name", ""),
            "time":      event.get("date", ""),
        })

    log.info(f"ESPN: {len(games)} NCAAB games on {date}")
    return games


def match_odds_to_espn(espn_games: list[dict], odds_games: list[dict]) -> list[dict]:
    """
    For each ESPN game, find the best-matching odds game and attach lines.
    Adds an 'odds' key to each ESPN game dict (None if no match found).
    """
    for eg in espn_games:
        ea = _clean(eg["away_team"])
        eh = _clean(eg["home_team"])
        best_match, best_score = None, 0.0

        for og in odds_games:
            oa = _clean(og["away_team"])
            oh = _clean(og["home_team"])

            def overlap(a, b):
                ta, tb = set(a.split()), set(b.split())
                return len(ta & tb) / max(len(ta), len(tb)) if ta and tb else 0.0

            score = overlap(ea, oa) + overlap(eh, oh)
            if score > best_score:
                best_score, best_match = score, og

        eg["odds"] = best_match if best_score >= 0.8 else None

    matched = sum(1 for g in espn_games if g["odds"])
    log.info(f"Odds matched for {matched}/{len(espn_games)} ESPN games")
    return espn_games


    """Fetch all upcoming NCAAB games with moneyline, spread, and total from The Odds API."""
    params = {
        "apiKey":      api_key,
        "regions":     "us",
        "markets":     "h2h,spreads,totals",
        "oddsFormat":  "american",
        "bookmakers":  "draftkings,fanduel,betmgm,caesars",
    }
    resp = requests.get(ODDS_API_BASE, params=params, timeout=15)
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining", "?")
    log.info(f"Odds API: {len(resp.json())} games fetched  (requests remaining: {remaining})")
    return resp.json()



# Cache of GID → full team name scraped from StatBroadcast
_GID_NAME_CACHE: dict[str, str] = {}

def _clean(s: str) -> str:
    """Normalize a team name for comparison."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    for noise in ["university", "college", "state", "the", "at", "of"]:
        s = re.sub(rf"\b{noise}\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def gid_to_full_name(gid: str, known_gids: list[dict]) -> str:
    """
    Return the full team name for a GID by looking it up in the StatBroadcast
    events page data (already scraped into known_gids as {gid: name} dicts).
    Falls back to the GID itself if not found.
    """
    if gid in _GID_NAME_CACHE:
        return _GID_NAME_CACHE[gid]

    # known_gids may be a list of gid strings or a dict of {gid: name}
    if isinstance(known_gids, dict):
        name = known_gids.get(gid, gid)
    else:
        name = gid  # just the GID string — we'll rely on fuzzy match

    _GID_NAME_CACHE[gid] = name
    return name


def find_game(games: list[dict], visitor_gid: str, home_gid: str,
              known_gids=None) -> dict | None:
    """
    Find the matching game in the odds feed using fuzzy team name matching.
    Tries multiple strategies in order:
      1. GID as substring of odds team name (e.g. 'hou' in 'houston')
      2. Full name lookup via known_gids
      3. Token overlap scoring as final fallback
    """
    vis_gid_clean  = _clean(visitor_gid)
    home_gid_clean = _clean(home_gid)

    # Get full names if available
    vis_full  = _clean(gid_to_full_name(visitor_gid, known_gids or {}))
    home_full = _clean(gid_to_full_name(home_gid,    known_gids or {}))

    def matches(query_gid: str, query_full: str, team_name: str) -> bool:
        t = _clean(team_name)
        # GID is substring of team name or vice versa
        if query_gid in t or t in query_gid:
            return True
        # Full name tokens — majority overlap
        q_tokens = set(query_full.split())
        t_tokens = set(t.split())
        if q_tokens and len(q_tokens & t_tokens) / len(q_tokens) >= 0.5:
            return True
        return False

    # Pass 1: strict match on both sides
    for game in games:
        away_n = game["away_team"]
        home_n = game["home_team"]
        if matches(vis_gid_clean, vis_full, away_n) and \
           matches(home_gid_clean, home_full, home_n):
            log.info(f"Matched: {away_n} @ {home_n}")
            return game

    # No match — log available games for debugging
    log.warning(
        f"Could not match '{visitor_gid}' vs '{home_gid}' in odds feed.\n"
        f"  Available:\n" +
        "\n".join(f"    {g['away_team']} @ {g['home_team']}" for g in games)
    )
    return None


def _load_name_gid_cache() -> dict[str, str]:
    """Load the persistent name→GID mapping cache."""
    if NAME_TO_GID_CACHE_FILE.exists():
        return json.loads(NAME_TO_GID_CACHE_FILE.read_text())
    return {}


def _save_name_gid_cache(cache: dict[str, str]) -> None:
    NAME_TO_GID_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    NAME_TO_GID_CACHE_FILE.write_text(json.dumps(cache, indent=2))


def odds_team_to_gid(team_name: str, known_gids: list[str],
                     interactive: bool = True) -> str | None:
    """
    Map an odds feed team name to a StatBroadcast GID.

    Resolution order:
      1. Persistent cache (game_xml_cache/name_to_gid.json)
      2. High-confidence fuzzy match (score >= 0.7) — auto-accepted
      3. Low-confidence — show top candidates and prompt user to pick
         (only in interactive mode; in non-interactive mode, skips)

    Confirmed matches are saved to the cache for future runs.
    """
    cache = _load_name_gid_cache()
    key   = team_name.strip().lower()

    # 1. Cache hit
    if key in cache:
        return cache[key]

    cleaned = _clean(team_name)
    tokens  = set(cleaned.split()) - {""}

    scored = []
    for gid in known_gids:
        gid_clean  = _clean(gid)
        gid_tokens = set(gid_clean.split()) - {""}

        # Substring match — strong signal
        if gid_clean in cleaned or cleaned in gid_clean:
            score = 0.95
        elif tokens and gid_tokens:
            overlap = len(tokens & gid_tokens)
            score   = overlap / max(len(tokens), len(gid_tokens))
        else:
            score = 0.0

        if score > 0:
            scored.append((score, gid))

    scored.sort(reverse=True)
    top = scored[:5]

    if not top:
        log.warning(f"No GID candidates found for '{team_name}'")
        return None

    best_score, best_gid = top[0]

    # 2. High-confidence auto-accept
    if best_score >= 0.7:
        log.info(f"  Auto-matched '{team_name}' → '{best_gid}' (score {best_score:.2f})")
        cache[key] = best_gid
        _save_name_gid_cache(cache)
        return best_gid

    # 3. Low-confidence — prompt user
    if not interactive:
        log.warning(f"  Uncertain match for '{team_name}' (best: '{best_gid}' @ {best_score:.2f}) — skipping")
        return None

    print(f"\n  ❓ Uncertain match for '{team_name}'. Top candidates:")
    for i, (score, gid) in enumerate(top, 1):
        print(f"     {i}. {gid:<20}  (score {score:.2f})")
    print(f"     s. Skip this team")

    while True:
        choice = input("  Select [1-5 / s]: ").strip().lower()
        if choice == "s":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(top):
            chosen_gid = top[int(choice) - 1][1]
            cache[key] = chosen_gid
            _save_name_gid_cache(cache)
            log.info(f"  Saved '{team_name}' → '{chosen_gid}' to cache")
            return chosen_gid
        print("  Invalid choice, try again.")




def extract_lines(game: dict, preferred_book: str | None = None) -> dict:
    """
    Extract moneyline, spread, and total from a game dict.

    If preferred_book is set (e.g. 'draftkings'), uses that book exclusively.
    Falls back to consensus average across all available books if the preferred
    book isn't offering lines for this game.
    """
    DEFAULT_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars",
                     "williamhill_us", "barstool", "pointsbetus", "espnbet",
                     "betonlineag", "bovada", "mybookieag", "lowvig"]

    bookmakers = game.get("bookmakers", [])

    # Try preferred book first
    if preferred_book:
        single = [bm for bm in bookmakers if bm["key"] == preferred_book.lower()]
        if single:
            log.info(f"  Using {preferred_book} lines")
            return _extract_from_bookmakers(game, single)
        else:
            available = [bm["key"] for bm in bookmakers]
            log.warning(f"  '{preferred_book}' not available for this game "
                        f"(available: {available}) — using consensus")

    # Consensus: use all available bookmakers (not just the big 4)
    return _extract_from_bookmakers(game, bookmakers)


def _extract_from_bookmakers(game: dict, bookmakers: list[dict]) -> dict:
    """Extract and average lines from a specific list of bookmakers."""
    h2h_vis, h2h_home         = [], []
    spread_line, vis_sp, home_sp = [], [], []
    total_line, over_o, under_o  = [], [], []

    for bm in bookmakers:
        for market in bm.get("markets", []):
            outcomes = {o["name"]: o for o in market.get("outcomes", [])}
            if market["key"] == "h2h":
                if game["away_team"] in outcomes:
                    h2h_vis.append(outcomes[game["away_team"]]["price"])
                if game["home_team"] in outcomes:
                    h2h_home.append(outcomes[game["home_team"]]["price"])
            elif market["key"] == "spreads":
                if game["away_team"] in outcomes:
                    o = outcomes[game["away_team"]]
                    spread_line.append(o["point"])
                    vis_sp.append(o["price"])
                if game["home_team"] in outcomes:
                    home_sp.append(outcomes[game["home_team"]]["price"])
            elif market["key"] == "totals":
                if "Over" in outcomes:
                    total_line.append(outcomes["Over"]["point"])
                    over_o.append(outcomes["Over"]["price"])
                if "Under" in outcomes:
                    under_o.append(outcomes["Under"]["price"])

    def avg(lst): return round(sum(lst) / len(lst)) if lst else None

    def mode_line(lst):
        if not lst:
            return None
        from collections import Counter
        return Counter(lst).most_common(1)[0][0]

    return {
        "visitor_ml":       avg(h2h_vis),
        "home_ml":          avg(h2h_home),
        "spread":           mode_line(spread_line),
        "vis_spread_odds":  avg(vis_sp),
        "home_spread_odds": avg(home_sp),
        "total":            mode_line(total_line),
        "over_odds":        avg(over_o),
        "under_odds":       avg(under_o),
    }


def print_fetched_lines(lines: dict, vis: str, home: str) -> None:
    print(f"\n{SEP}")
    print(f"  FETCHED LINES  —  {vis} (V) @ {home} (H)  [consensus across books]")
    print(SEP)
    print(f"  Moneyline:  {vis} {lines['visitor_ml']:+d}  /  {home} {lines['home_ml']:+d}")
    print(f"  Spread:     {vis} {lines['spread']:+.1f}  (odds {lines['vis_spread_odds']:+d} / {lines['home_spread_odds']:+d})")
    print(f"  Total:      {lines['total']}  (over {lines['over_odds']:+d} / under {lines['under_odds']:+d})")



def _sb_headers(referer: str) -> dict:
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer":    referer,
        "X-Requested-With": "XMLHttpRequest",
    }


def load_or_scrape_gids(force_refresh: bool = False) -> list[str]:
    """
    Return the full list of known StatBroadcast team GIDs.
    Scrapes the events page on first run, then caches to disk.
    """
    if not force_refresh and GID_CACHE_FILE.exists():
        gids = json.loads(GID_CACHE_FILE.read_text())
        log.info(f"Loaded {len(gids)} team GIDs from cache ({GID_CACHE_FILE})")
        return gids

    log.info(f"Scraping team GIDs from {SB_EVENTS} …")
    resp = requests.get(SB_EVENTS, timeout=30)
    resp.raise_for_status()

    soup     = BeautifulSoup(resp.text, "html.parser")
    skip     = {"bcs","stat","admin","system","index","demo","support","blog","about"}
    gids     = set()

    for link in soup.find_all("a", href=True):
        m = re.search(r"[?&]gid=([a-zA-Z0-9]+)", link["href"])
        if m:
            gid = m.group(1).lower()
            if gid not in skip and len(gid) >= 2:
                gids.add(gid)

    gids = sorted(gids)
    GID_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    GID_CACHE_FILE.write_text(json.dumps(gids, indent=2))
    log.info(f"Found and cached {len(gids)} team GIDs → {GID_CACHE_FILE}")
    return gids


def resolve_gid(query: str, known_gids: list[str]) -> str:
    """
    Resolve a user-supplied team name/GID to an actual StatBroadcast GID.

    Strategy (in order):
      1. Exact match
      2. Prefix match  (e.g. "duke" matches "dukembb")
      3. Substring match
      4. Abort with helpful error listing close candidates
    """
    q = query.lower().strip()

    # 1. Exact
    if q in known_gids:
        log.info(f"  Resolved '{query}' → '{q}' (exact match)")
        return q

    # 2. Prefix
    prefix_matches = [g for g in known_gids if g.startswith(q)]
    if len(prefix_matches) == 1:
        log.info(f"  Resolved '{query}' → '{prefix_matches[0]}' (prefix match)")
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        # Pick shortest (most canonical) prefix match
        best = min(prefix_matches, key=len)
        log.info(f"  Resolved '{query}' → '{best}' (shortest of {prefix_matches})")
        return best

    # 3. Substring
    sub_matches = [g for g in known_gids if q in g]
    if len(sub_matches) == 1:
        log.info(f"  Resolved '{query}' → '{sub_matches[0]}' (substring match)")
        return sub_matches[0]
    if len(sub_matches) > 1:
        best = min(sub_matches, key=len)
        log.info(f"  Resolved '{query}' → '{best}' (shortest of substring matches {sub_matches})")
        return best

    # 4. Fail helpfully
    # Show any GID containing any character sequence overlap
    suggestions = [g for g in known_gids if any(q[i:i+3] in g for i in range(len(q)-2))][:10]
    raise SystemExit(
        f"\nERROR: Could not resolve '{query}' to a known StatBroadcast GID.\n"
        f"  Possible matches: {suggestions or '(none found)'}\n"
        f"  Run with --list-gids to see all known GIDs.\n"
        f"  Run with --refresh-gids to re-scrape the GID list.\n"
    )


def get_game_ids_for_team(team_gid: str, n_games: int = 5) -> list[str]:
    """
    Return the most recent *n_games* game IDs for a StatBroadcast team GID.

    Delegates to get_game_ids.js (Node/Puppeteer) which loads the archive
    page in a real browser, waits for #archiveTable to populate, and returns
    a JSON array of game ID strings via stdout.
    """
    import subprocess

    # Look for the JS script next to this Python file, or in cwd
    script = Path(__file__).parent / "get_game_ids.js"
    if not script.exists():
        script = Path("get_game_ids.js")
    if not script.exists():
        raise RuntimeError(
            "get_game_ids.js not found. Place it in the same directory as this script.\n"
            "Install deps with: npm install puppeteer"
        )

    log.info(f"[{team_gid}] Running: node {script} {team_gid} {n_games}")
    result = subprocess.run(
        ["node", str(script), team_gid, str(n_games)],
        capture_output=True, text=True, timeout=60
    )

    # Stderr is Puppeteer's progress logs — forward them
    for line in result.stderr.strip().splitlines():
        log.info(f"  [node] {line}")

    if result.returncode != 0:
        raise RuntimeError(
            f"get_game_ids.js exited with code {result.returncode} for '{team_gid}'"
        )

    try:
        game_ids = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Could not parse game IDs JSON from node output: {result.stdout[:200]}"
        )

    log.info(f"[{team_gid}] Got {len(game_ids)} game IDs: {game_ids}")
    return game_ids


def fetch_game_xml(game_id: str, cache_dir: Path | None = None) -> str:
    """Download (or load from cache) a StatBroadcast game XML.
    
    Uses the archive subdomain: http://archive.statbroadcast.com/<id>.xml
    """
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"{game_id}.xml"
        if cached.exists():
            log.info(f"  [cache] {cached}")
            return cached.read_text()

    url = f"http://archive.statbroadcast.com/{game_id}.xml"
    log.info(f"  Downloading game {game_id} from {url}")
    resp = requests.get(url, timeout=30, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    })
    resp.raise_for_status()
    xml_text = resp.text

    if cache_dir:
        cached.write_text(xml_text)
        log.info(f"  Cached → {cached}")

    return xml_text


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  XML → TRANSITION COUNT MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def play_to_state(p: dict) -> int | None:
    action = p.get("action", "")
    ptype  = p.get("type",   "")
    vh     = p.get("vh",     "")
    prefix = "V" if vh == "V" else "H"
    if action == "GOOD":
        if "3PTR" in ptype: return STATE_IDX[f"{prefix}_made_3"]
        if "FT"   in ptype: return STATE_IDX[f"{prefix}_made_ft"]
        return STATE_IDX[f"{prefix}_made_2"]
    if action == "MISS":
        if "3PTR" in ptype: return STATE_IDX[f"{prefix}_miss_3"]
        if "FT"   in ptype: return STATE_IDX[f"{prefix}_miss_ft"]
        return STATE_IDX[f"{prefix}_miss_2"]
    if action == "TURNOVER":
        return STATE_IDX[f"{prefix}_turnover"]
    return None


def xml_to_counts(xml_text: str) -> tuple[np.ndarray, int, int | None, int | None, str, str]:
    """
    Parse a DakStats bbgame XML string.

    Returns
    -------
    counts      : (N_STATES, N_STATES) raw transition count matrix
    n_scoring   : number of scoring events (used as n_possessions)
    actual_v    : visitor final score (or None)
    actual_h    : home final score (or None)
    vis_name    : visitor team name
    home_name   : home team name
    """
    root = ET.fromstring(xml_text)

    vis_name  = "Visitor"
    home_name = "Home"
    venue = root.find(".//venue")
    if venue is not None:
        vis_name  = venue.get("visname",  vis_name)
        home_name = venue.get("homename", home_name)

    plays = [pl.attrib for pl in root.iter("play")]

    counts = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    prev   = None
    for p in plays:
        s = play_to_state(p)
        if s is None:
            continue
        if prev is not None:
            counts[prev, s] += 1
        prev = s

    n_scoring = int(counts.sum())

    actual_v = actual_h = None
    for p in reversed(plays):
        if "vscore" in p and "hscore" in p:
            try:
                actual_v = int(p["vscore"])
                actual_h = int(p["hscore"])
            except ValueError:
                pass
            break

    return counts, n_scoring, actual_v, actual_h, vis_name, home_name


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  BLEND MATRICES
# ═══════════════════════════════════════════════════════════════════════════════

def counts_to_prob(counts: np.ndarray) -> np.ndarray:
    """Normalize a raw count matrix into a probability matrix."""
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = counts / row_sums
    # Uniform fallback for zero rows
    zero_rows = P.sum(axis=1) == 0
    P[zero_rows] = 1.0 / N_STATES
    return P


def blend_team_matrices(
    game_xmls: list[str],
    weighting: str = "recency",   # "equal" | "recency"
) -> tuple[np.ndarray, float, str, str]:
    """
    Build a single blended transition probability matrix from multiple game XMLs.

    Games are assumed to be ordered most-recent first.
    With weighting='recency', game i (0=most recent) gets weight 2^(-i).
    With weighting='equal', raw counts are simply summed.

    Returns
    -------
    P           : blended (N_STATES, N_STATES) probability matrix
    avg_poss    : average number of possessions across games
    vis_name    : last seen visitor team name
    home_name   : last seen home team name
    """
    accumulated = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    total_poss  = 0
    vis_name    = "Visitor"
    home_name   = "Home"

    n = len(game_xmls)
    if weighting == "recency":
        weights = np.array([2.0 ** (-i) for i in range(n)])
        weights /= weights.sum()          # normalise so they sum to 1
    else:
        weights = np.ones(n) / n

    for i, xml_text in enumerate(game_xmls):
        try:
            counts, n_scoring, _, _, vn, hn = xml_to_counts(xml_text)
        except Exception as e:
            log.warning(f"  Skipping game {i+1}: {e}")
            continue

        accumulated  += weights[i] * counts
        total_poss   += n_scoring
        vis_name      = vn
        home_name     = hn

    P        = counts_to_prob(accumulated)
    avg_poss = total_poss / max(len(game_xmls), 1)

    return P, avg_poss, vis_name, home_name


def combine_visitor_home_matrices(
    P_vis: np.ndarray,
    P_home: np.ndarray,
) -> np.ndarray:
    """
    Combine visitor and home matrices into one matchup matrix.

    Strategy: simple average.  The visitor matrix carries the visitor team's
    scoring tendencies; the home matrix carries the home team's.  Averaging
    preserves both while modeling the matchup as a blend of each side's
    historical play patterns.
    """
    return (P_vis + P_home) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MCMC SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _run_chain(P: np.ndarray, n_sim: int, n_steps: int,
               pts_vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized Markov chain; returns total-score array of shape (n_sim,)."""
    P_cum  = np.cumsum(P, axis=1)
    draws  = rng.random((n_steps, n_sim))
    states = rng.integers(0, N_STATES, size=n_sim)
    scores = np.zeros(n_sim, dtype=np.int32)

    for step in range(n_steps):
        states = (draws[step, :, np.newaxis] > P_cum[states]).sum(axis=1)
        states = np.clip(states, 0, N_STATES - 1)
        scores += pts_vec[states]

    return scores


def run_simulation(P: np.ndarray, n_sim: int = 50_000,
                   n_possessions: int = 150, seed: int = 42) -> dict:
    """
    Run the MCMC simulation and return a results dict.

    n_possessions is the total number of scoring events per game.
    Both visitor and home chains run for the full n_possessions steps —
    the matrix encodes which team scores on each transition, so both
    chains must see all events to accumulate correct totals.
    """
    rng      = np.random.default_rng(seed)
    # Run a single chain; V_PTS and H_PTS extract each team's contribution
    scores_v = _run_chain(P, n_sim, n_possessions, V_PTS, rng)
    scores_h = _run_chain(P, n_sim, n_possessions, H_PTS, rng)

    # Sanity check — warn if avg total looks implausibly low
    avg_total = float((scores_v + scores_h).mean())
    if avg_total < 80:
        log.warning(
            f"Avg simulated total is {avg_total:.1f} — suspiciously low. "
            f"Check n_possessions ({n_possessions}) and transition matrix."
        )

    spread = scores_v - scores_h
    totals = scores_v + scores_h

    v_wins = int((scores_v > scores_h).sum())
    h_wins = int((scores_h > scores_v).sum())

    return {
        "v_scores":        scores_v,
        "h_scores":        scores_h,
        "spread":          spread,
        "totals":          totals,
        "n_sim":           n_sim,
        "visitor_win_pct": v_wins / n_sim,
        "home_win_pct":    h_wins / n_sim,
        "tie_pct":         (n_sim - v_wins - h_wins) / n_sim,
        "avg_visitor":     float(scores_v.mean()),
        "avg_home":        float(scores_h.mean()),
        "avg_spread":      float(spread.mean()),
        "std_spread":      float(spread.std()),
        "avg_total":       float(totals.mean()),
        "std_total":       float(totals.std()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  BETTING QUERY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def american_to_implied(odds: float) -> float:
    """Convert American odds to implied (break-even) probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def decimal_to_implied(odds: float) -> float:
    return 1.0 / odds


def implied_to_american(prob: float) -> str:
    """Convert a probability to American odds string."""
    if prob <= 0: return "0"
    if prob >= 1: return "−∞"
    if prob >= 0.5:
        return f"{-round(prob / (1 - prob) * 100)}"
    else:
        return f"+{round((1 - prob) / prob * 100)}"


def edge(model_prob: float, book_implied: float) -> float:
    """Kelly-style edge: positive means model favours the bet."""
    return model_prob - book_implied


def kelly_fraction(model_prob: float, american_odds: float,
                   fraction: float = 1.0) -> float:
    """
    Full Kelly criterion stake as a fraction of bankroll, scaled by fraction.

    f* = (b*p - q) / b   where b = net decimal odds (profit per unit staked)

    Parameters
    ----------
    model_prob    : model's estimated win probability
    american_odds : book's American odds for this side
    fraction      : Kelly multiplier, e.g. 0.25 = quarter Kelly

    Returns
    -------
    Recommended bet as a fraction of bankroll (0 if no edge).
    """
    if american_odds > 0:
        b = american_odds / 100.0
    else:
        b = 100.0 / abs(american_odds)

    p = model_prob
    q = 1.0 - p
    f = (b * p - q) / b

    return max(0.0, f * fraction)


def _kelly_str(model_prob: float, american_odds: float | None,
               fraction: float) -> str:
    """Return a formatted Kelly stake string, or empty string if no odds."""
    if american_odds is None:
        return ""
    k = kelly_fraction(model_prob, american_odds, fraction)
    if k <= 0:
        return "  (no edge — do not bet)"
    return f"  Kelly stake: {k*100:.2f}% of bankroll"


def query_total(
    res: dict,
    line: float,
    over_odds: float | None = None,
    under_odds: float | None = None,
    odds_format: str = "american",
) -> dict:
    """
    Query the over/under distribution against a total line.

    Parameters
    ----------
    line        : the book's total (e.g. 155.5)
    over_odds   : American (or decimal) odds for the OVER bet
    under_odds  : American (or decimal) odds for the UNDER bet
    odds_format : "american" | "decimal"
    """
    totals = res["totals"]
    n      = res["n_sim"]

    model_over  = float((totals > line).sum()) / n
    model_under = float((totals < line).sum()) / n
    model_push  = 1.0 - model_over - model_under

    result = {
        "line":        line,
        "model_over":  model_over,
        "model_under": model_under,
        "model_push":  model_push,
        "model_over_american":  implied_to_american(model_over),
        "model_under_american": implied_to_american(model_under),
    }

    _parse = american_to_implied if odds_format == "american" else decimal_to_implied

    if over_odds is not None:
        book_implied = _parse(over_odds)
        result["book_over_implied"] = book_implied
        result["over_edge"]         = edge(model_over, book_implied)
        result["_over_odds_raw"]    = over_odds

    if under_odds is not None:
        book_implied = _parse(under_odds)
        result["book_under_implied"] = book_implied
        result["under_edge"]         = edge(model_under, book_implied)
        result["_under_odds_raw"]    = under_odds

    return result


def query_spread(
    res: dict,
    line: float,
    visitor_odds: float | None = None,
    home_odds:    float | None = None,
    odds_format:  str = "american",
) -> dict:
    """
    Query the spread distribution.

    Parameters
    ----------
    line         : spread from the visitor's perspective (e.g. -4.5 means
                   visitor is favoured by 4.5; visitor must win by 5+)
    visitor_odds : odds for betting the visitor to cover
    home_odds    : odds for betting the home team to cover (take the points)
    """
    spread = res["spread"]   # visitor − home
    n      = res["n_sim"]

    model_vis_cover  = float((spread > -line).sum()) / n   # visitor covers if spread > -line
    model_home_cover = float((spread < -line).sum()) / n
    model_push       = 1.0 - model_vis_cover - model_home_cover

    result = {
        "line":                line,
        "model_visitor_cover": model_vis_cover,
        "model_home_cover":    model_home_cover,
        "model_push":          model_push,
        "model_vis_american":  implied_to_american(model_vis_cover),
        "model_home_american": implied_to_american(model_home_cover),
    }

    _parse = american_to_implied if odds_format == "american" else decimal_to_implied

    if visitor_odds is not None:
        book_implied = _parse(visitor_odds)
        result["book_visitor_implied"]  = book_implied
        result["visitor_edge"]          = edge(model_vis_cover, book_implied)
        result["_vis_spread_odds_raw"]  = visitor_odds

    if home_odds is not None:
        book_implied = _parse(home_odds)
        result["book_home_implied"]      = book_implied
        result["home_edge"]              = edge(model_home_cover, book_implied)
        result["_home_spread_odds_raw"]  = home_odds

    return result


def query_moneyline(
    res: dict,
    visitor_ml: float | None = None,
    home_ml:    float | None = None,
    odds_format: str = "american",
) -> dict:
    """Query the win-probability distribution against moneyline odds."""
    result = {
        "model_visitor_win": res["visitor_win_pct"],
        "model_home_win":    res["home_win_pct"],
        "model_tie":         res["tie_pct"],
        "model_vis_ml":      implied_to_american(res["visitor_win_pct"]),
        "model_home_ml":     implied_to_american(res["home_win_pct"]),
    }

    _parse = american_to_implied if odds_format == "american" else decimal_to_implied

    if visitor_ml is not None:
        book_implied = _parse(visitor_ml)
        result["book_visitor_implied"] = book_implied
        result["visitor_ml_edge"]      = edge(res["visitor_win_pct"], book_implied)
        result["_visitor_ml_raw"]      = visitor_ml

    if home_ml is not None:
        book_implied = _parse(home_ml)
        result["book_home_implied"] = book_implied
        result["home_ml_edge"]      = edge(res["home_win_pct"], book_implied)
        result["_home_ml_raw"]      = home_ml

    return result


def query_custom(res: dict, variable: str, direction: str, threshold: float) -> dict:
    """
    Generic query: P(variable OP threshold).

    Parameters
    ----------
    variable  : "total" | "spread" | "visitor_score" | "home_score"
    direction : "over" | "under" | "at_least" | "at_most" | "exactly"
    threshold : numeric value
    """
    mapping = {
        "total":         res["totals"],
        "spread":        res["spread"],
        "visitor_score": res["v_scores"],
        "home_score":    res["h_scores"],
    }
    arr = mapping[variable]
    n   = res["n_sim"]

    if direction in ("over", "at_least", "above"):
        prob = float((arr >= threshold).sum()) / n
        op   = ">="
    elif direction in ("under", "at_most", "below"):
        prob = float((arr <= threshold).sum()) / n
        op   = "<="
    elif direction == "exactly":
        prob = float((arr == int(threshold)).sum()) / n
        op   = "=="
    else:
        raise ValueError(f"Unknown direction '{direction}'. Use over/under/exactly.")

    return {
        "variable":  variable,
        "op":        op,
        "threshold": threshold,
        "prob":      prob,
        "american":  implied_to_american(prob),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  PRETTY PRINTING
# ═══════════════════════════════════════════════════════════════════════════════

SEP = "─" * 60

def _pct(p: float) -> str: return f"{p*100:.1f}%"
def _edge_str(e: float) -> str:
    arrow = "▲" if e > 0 else "▼"
    return f"{arrow} {abs(e)*100:.1f}pp edge {'FOR' if e > 0 else 'AGAINST'} bet"


def print_simulation_summary(res: dict, vis_name: str, home_name: str) -> None:
    print(f"\n{SEP}")
    print(f"  MCMC SIMULATION  —  {res['n_sim']:,} runs")
    print(f"  {vis_name} (V)  vs  {home_name} (H)")
    print(SEP)
    print(f"  Win probability:   {vis_name} {_pct(res['visitor_win_pct'])}  |  "
          f"{home_name} {_pct(res['home_win_pct'])}  |  Tie {_pct(res['tie_pct'])}")
    print(f"  Avg scores:        {vis_name} {res['avg_visitor']:.1f}  |  {home_name} {res['avg_home']:.1f}")
    print(f"  Over/Under:        mean {res['avg_total']:.1f}  σ={res['std_total']:.1f}")
    print(f"  Spread (V−H):      mean {res['avg_spread']:+.1f}  σ={res['std_spread']:.1f}")
    print()
    # Quick percentile table
    for label, arr in [("Total", res["totals"]), ("Spread", res["spread"])]:
        pcts = np.percentile(arr, [5, 25, 50, 75, 95])
        print(f"  {label} percentiles:  "
              f"5th={pcts[0]:+.0f}  25th={pcts[1]:+.0f}  "
              f"med={pcts[2]:+.0f}  75th={pcts[3]:+.0f}  95th={pcts[4]:+.0f}")


def print_total_query(q: dict, kelly: float = 1.0) -> None:
    print(f"\n{SEP}")
    print(f"  OVER / UNDER  —  Line: {q['line']}")
    print(SEP)
    print(f"  Model OVER  probability: {_pct(q['model_over'])}  "
          f"(fair odds {q['model_over_american']})")
    print(f"  Model UNDER probability: {_pct(q['model_under'])}  "
          f"(fair odds {q['model_under_american']})")
    print(f"  Model push  probability: {_pct(q['model_push'])}")

    if "book_over_implied" in q:
        print(f"\n  Book OVER  implied: {_pct(q['book_over_implied'])}  →  {_edge_str(q['over_edge'])}")
        print(_kelly_str(q['model_over'], q.get('_over_odds_raw'), kelly))
    if "book_under_implied" in q:
        print(f"  Book UNDER implied: {_pct(q['book_under_implied'])}  →  {_edge_str(q['under_edge'])}")
        print(_kelly_str(q['model_under'], q.get('_under_odds_raw'), kelly))


def print_spread_query(q: dict, vis_name: str, home_name: str, kelly: float = 1.0) -> None:
    line_str = f"{q['line']:+.1f}"
    print(f"\n{SEP}")
    print(f"  SPREAD  —  {vis_name} {line_str}  /  {home_name} {-q['line']:+.1f}")
    print(SEP)
    print(f"  Model {vis_name} covers: {_pct(q['model_visitor_cover'])}  "
          f"(fair {q['model_vis_american']})")
    print(f"  Model {home_name} covers: {_pct(q['model_home_cover'])}  "
          f"(fair {q['model_home_american']})")
    print(f"  Model push:            {_pct(q['model_push'])}")

    if "book_visitor_implied" in q:
        print(f"\n  Book {vis_name} implied: {_pct(q['book_visitor_implied'])}  →  "
              f"{_edge_str(q['visitor_edge'])}")
        print(_kelly_str(q['model_visitor_cover'], q.get('_vis_spread_odds_raw'), kelly))
    if "book_home_implied" in q:
        print(f"  Book {home_name} implied:  {_pct(q['book_home_implied'])}  →  "
              f"{_edge_str(q['home_edge'])}")
        print(_kelly_str(q['model_home_cover'], q.get('_home_spread_odds_raw'), kelly))


def print_ml_query(q: dict, vis_name: str, home_name: str, kelly: float = 1.0) -> None:
    print(f"\n{SEP}")
    print(f"  MONEYLINE")
    print(SEP)
    print(f"  Model {vis_name} win: {_pct(q['model_visitor_win'])}  "
          f"(fair {q['model_vis_ml']})")
    print(f"  Model {home_name} win:  {_pct(q['model_home_win'])}  "
          f"(fair {q['model_home_ml']})")

    if "book_visitor_implied" in q:
        print(f"\n  Book {vis_name} implied: {_pct(q['book_visitor_implied'])}  →  "
              f"{_edge_str(q['visitor_ml_edge'])}")
        print(_kelly_str(q['model_visitor_win'], q.get('_visitor_ml_raw'), kelly))
    if "book_home_implied" in q:
        print(f"  Book {home_name} implied:  {_pct(q['book_home_implied'])}  →  "
              f"{_edge_str(q['home_ml_edge'])}")
        print(_kelly_str(q['model_home_win'], q.get('_home_ml_raw'), kelly))


def print_custom_query(q: dict) -> None:
    print(f"\n  Custom query — P({q['variable']} {q['op']} {q['threshold']})  "
          f"=  {_pct(q['prob'])}  (fair odds {q['american']})")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(
    res: dict,
    vis_name:  str,
    home_name: str,
    out_path:  str = "simulation.png",
    total_line:  float | None = None,
    spread_line: float | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    v_scores = res["v_scores"]
    h_scores = res["h_scores"]
    spread   = res["spread"]
    totals   = res["totals"]
    n_sim    = res["n_sim"]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"MCMC Game Simulation  —  {vis_name} (V) vs {home_name} (H)  "
        f"({n_sim:,} simulations)",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    ax_scores  = fig.add_subplot(gs[0, 0])
    ax_win     = fig.add_subplot(gs[0, 1])
    ax_spread  = fig.add_subplot(gs[0, 2])
    ax_total   = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[1, 1])
    ax_stats   = fig.add_subplot(gs[1, 2])
    ax_stats.axis("off")

    lo   = min(v_scores.min(), h_scores.min()) - 2
    hi   = max(v_scores.max(), h_scores.max()) + 4
    bins = np.arange(lo, hi, 2)

    # Score distributions
    ax_scores.hist(v_scores, bins=bins, alpha=0.6, color="royalblue", label=f"{vis_name} (V)")
    ax_scores.hist(h_scores, bins=bins, alpha=0.6, color="tomato",    label=f"{home_name} (H)")
    ax_scores.set_title("Score Distributions"); ax_scores.set_xlabel("Points")
    ax_scores.legend(fontsize=8)

    # Win probability
    labels = [f"{vis_name}", f"{home_name}", "Tie"]
    vals   = [res["visitor_win_pct"], res["home_win_pct"], res["tie_pct"]]
    bars   = ax_win.bar(labels, vals, color=["royalblue","tomato","silver"], edgecolor="white")
    ax_win.set_title("Win Probability"); ax_win.set_ylim(0, 1)
    for bar, val in zip(bars, vals):
        ax_win.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val*100:.1f}%", ha="center", fontweight="bold", fontsize=9)

    # Spread
    ax_spread.hist(spread, bins=60, color="mediumpurple", edgecolor="white", lw=0.3)
    ax_spread.axvline(0, color="black", lw=1.5, label="Pick'em")
    ax_spread.axvline(res["avg_spread"], color="gold", lw=2, ls="--",
                      label=f"Mean: {res['avg_spread']:+.1f}")
    if spread_line is not None:
        ax_spread.axvline(-spread_line, color="cyan", lw=2, ls=":",
                          label=f"Book line: {-spread_line:+.1f}")
    ax_spread.set_title(f"Spread  ({vis_name} − {home_name})")
    ax_spread.set_xlabel("Points"); ax_spread.legend(fontsize=8)

    # Over/Under
    ax_total.hist(totals, bins=60, color="darkorange", edgecolor="white", lw=0.3)
    ax_total.axvline(res["avg_total"], color="gold", lw=2, ls="--",
                     label=f"Mean: {res['avg_total']:.1f}")
    if total_line is not None:
        ax_total.axvline(total_line, color="cyan", lw=2, ls=":",
                         label=f"Book O/U: {total_line}")
        # Shade over/under regions
        over_pct  = (totals > total_line).mean()
        under_pct = (totals < total_line).mean()
        ax_total.fill_betweenx([0, ax_total.get_ylim()[1] or 1],
                                total_line, totals.max() + 5,
                                alpha=0.08, color="green", label=f"Over {over_pct*100:.0f}%")
        ax_total.fill_betweenx([0, ax_total.get_ylim()[1] or 1],
                                totals.min() - 5, total_line,
                                alpha=0.08, color="red",   label=f"Under {under_pct*100:.0f}%")
    ax_total.set_title("Over/Under  (total points)")
    ax_total.set_xlabel("Points"); ax_total.legend(fontsize=7)

    # Scatter
    rng_plot = np.random.default_rng(0)
    sample   = rng_plot.choice(n_sim, min(2000, n_sim), replace=False)
    ax_scatter.scatter(v_scores[sample], h_scores[sample], alpha=0.08, s=6, color="steelblue")
    ax_scatter.plot([lo, hi], [lo, hi], "k--", lw=1, label="Equal score")
    ax_scatter.set_xlabel(f"{vis_name} score")
    ax_scatter.set_ylabel(f"{home_name} score")
    ax_scatter.set_title("Score Scatter (sample)")
    ax_scatter.legend(fontsize=8)

    # Stats panel
    sp = np.percentile(spread, [10, 25, 50, 75, 90])
    tp = np.percentile(totals, [10, 25, 50, 75, 90])
    lines_txt = [
        f"── Simulation Summary ────────",
        f"{vis_name} avg:   {res['avg_visitor']:.1f}",
        f"{home_name} avg:  {res['avg_home']:.1f}",
        "",
        f"Spread line:  {res['avg_spread']:+.1f}  (σ {res['std_spread']:.1f})",
        f"  median:     {sp[2]:+.0f}",
        f"  80% CI:     [{sp[1]:+.0f}, {sp[3]:+.0f}]",
        "",
        f"Over/Under:   {res['avg_total']:.1f}  (σ {res['std_total']:.1f})",
        f"  median:     {tp[2]:.0f}",
        f"  80% CI:     [{tp[1]:.0f}, {tp[3]:.0f}]",
    ]
    if total_line:
        lines_txt += ["", f"Book O/U line: {total_line}",
                      f"  Model over:  {(totals > total_line).mean()*100:.1f}%"]
    if spread_line:
        lines_txt += [f"Book spread:   {spread_line:+.1f}",
                      f"  Vis covers:  {(spread > -spread_line).mean()*100:.1f}%"]

    ax_stats.text(0.05, 0.95, "\n".join(lines_txt), transform=ax_stats.transAxes,
                  fontsize=9, va="top", fontfamily="monospace",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"Plot saved → '{out_path}'")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Basketball MCMC betting simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Teams (not required in --screen mode)
    p.add_argument("--visitor", metavar="GID", default=None,
                   help="Visitor team StatBroadcast GID (e.g. duke)")
    p.add_argument("--home",    metavar="GID", default=None,
                   help="Home team StatBroadcast GID (e.g. unc)")

    # Odds API
    p.add_argument("--odds-api-key", metavar="KEY", default=None,
                   help="The Odds API key — auto-fetches lines (get free key at the-odds-api.com)")
    p.add_argument("--book", metavar="BOOK", default=None,
                   help="Preferred bookmaker, e.g. draftkings, fanduel, betmgm, caesars. "
                        "Falls back to consensus if book not available for a game.")

    # Screener
    p.add_argument("--screen", action="store_true",
                   help="Screen all games today, rank by edge (requires --odds-api-key)")
    p.add_argument("--date", metavar="YYYY-MM-DD", default=None,
                   help="Date to screen, e.g. 2026-03-07 (default: today)")
    p.add_argument("--top", type=int, default=10,
                   help="Number of top bets to show in screen mode (default: 10)")
    p.add_argument("--markets", nargs="+",
                   choices=["ml", "spread", "total"], default=["ml", "spread", "total"],
                   help="Markets to evaluate, e.g. --markets spread total (default: all)")
    p.add_argument("--no-interactive", action="store_true",
                   help="Skip uncertain GID matches instead of prompting (for unattended runs)")
    p.add_argument("--sort", choices=["edge", "time"], default="edge",
                   help="Sort screener results by edge (default) or game time")
    p.add_argument("--games", nargs="+", metavar="ABBR",
                   help="Only simulate specific games by ESPN abbreviation, "
                        "e.g. --games 'DRKE VS BEL' 'FGCU VS LIP'")

    # Data source
    p.add_argument("--n-games",  type=int, default=5,
                   help="Number of recent games to blend (default: 5)")
    p.add_argument("--xml-dir",  metavar="DIR", default=None,
                   help="Directory for caching / pre-downloaded XMLs")
    p.add_argument("--weighting", choices=["equal", "recency"], default="recency",
                   help="How to weight games (default: recency — recent games weighted higher)")

    # Simulation
    p.add_argument("--nsim",    type=int, default=50_000, help="Simulations (default: 50000)")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--n-poss",  type=int, default=None,
                   help="Override possessions per simulation (default: avg of scraped games)")

    # Betting lines
    p.add_argument("--total",       type=float, default=None, help="O/U total line, e.g. 155.5")
    p.add_argument("--over-odds",   type=float, default=None, help="Odds for OVER (American)")
    p.add_argument("--under-odds",  type=float, default=None, help="Odds for UNDER (American)")
    p.add_argument("--spread",      type=float, default=None,
                   help="Spread from visitor perspective, e.g. -4.5")
    p.add_argument("--vis-spread-odds",  type=float, default=None,
                   help="Odds for visitor to cover spread")
    p.add_argument("--home-spread-odds", type=float, default=None,
                   help="Odds for home to cover spread")
    p.add_argument("--visitor-ml",  type=float, default=None, help="Visitor moneyline (American)")
    p.add_argument("--home-ml",     type=float, default=None, help="Home moneyline (American)")
    p.add_argument("--odds-format", choices=["american", "decimal"], default="american")

    # Custom query
    p.add_argument("--query-variable",  choices=["total","spread","visitor_score","home_score"],
                   default=None)
    p.add_argument("--query-direction", choices=["over","under","exactly"], default="over")
    p.add_argument("--query-threshold", type=float, default=None)

    # Output
    p.add_argument("--plot",    nargs="?", const="simulation.png", metavar="FILE",
                   help="Save plot PNG")
    p.add_argument("--json",    metavar="FILE", default=None,
                   help="Save full results as JSON")
    p.add_argument("--kelly", type=float, default=1.0,
                   help="Kelly fraction, e.g. 0.25 = quarter Kelly (default: 1.0 = full Kelly)")
    p.add_argument("--rate-limit", type=float, default=1.0,
                   help="Seconds between HTTP requests (default: 1.0)")
    p.add_argument("--list-gids", action="store_true",
                   help="Print all known team GIDs and exit")
    p.add_argument("--refresh-gids", action="store_true",
                   help="Re-scrape the GID list even if cached")

    return p


def simulate_game_for_screen(
    vis_gid: str, home_gid: str, lines: dict,
    known_gids: list[str], xml_cache: Path,
    n_games: int, weighting: str, nsim: int, kelly: float,
    markets: list[str] | None = None,
    game_label: str | None = None,
) -> list[dict]:
    """
    Run the full pipeline for one game and return a list of bet opportunity
    dicts, each with keys: game, market, side, model_prob, book_implied, edge, kelly.
    Returns empty list if GIDs can't be resolved or scraping fails.
    """
    markets = set(markets or ["ml", "spread", "total"])
    try:
        vg = resolve_gid(vis_gid, known_gids)
        hg = resolve_gid(home_gid, known_gids)
    except SystemExit:
        log.warning(f"  Skipping {vis_gid} @ {home_gid} — GID not found")
        return []

    try:
        vis_xmls  = _load_xmls(vg, n_games, xml_cache)
        home_xmls = _load_xmls(hg, n_games, xml_cache)
    except Exception as e:
        log.warning(f"  Skipping {vg} @ {hg} — scrape failed: {e}")
        return []

    if not vis_xmls or not home_xmls:
        return []

    P_vis,  avg_poss_vis,  _, _ = blend_team_matrices(vis_xmls,  weighting)
    P_home, avg_poss_home, _, _ = blend_team_matrices(home_xmls, weighting)
    P_match  = combine_visitor_home_matrices(P_vis, P_home)
    n_poss   = int((avg_poss_vis + avg_poss_home) / 2) or 150
    res      = run_simulation(P_match, n_sim=nsim, n_possessions=n_poss)

    game_label = game_label or f"{vg.upper()} @ {hg.upper()}"
    bets = []

    _parse = american_to_implied

    def _add(market, side, model_p, odds_raw):
        if odds_raw is None or model_p is None:
            return
        book_imp = _parse(odds_raw)
        e        = edge(model_p, book_imp)
        k        = kelly_fraction(model_p, odds_raw, kelly)
        bets.append({
            "game":         game_label,
            "market":       market,
            "side":         side,
            "model_prob":   model_p,
            "book_implied": book_imp,
            "edge":         e,
            "kelly":        k,
            "odds":         odds_raw,
        })

    # Moneyline
    if "ml" in markets:
        _add("ML", f"{vg.upper()} ML", res["visitor_win_pct"], lines.get("visitor_ml"))
        _add("ML", f"{hg.upper()} ML", res["home_win_pct"],    lines.get("home_ml"))

    # Spread
    if "spread" in markets and lines.get("spread") is not None:
        vis_line  = lines["spread"]        # e.g. +3.8 means visitor is underdog
        home_line = -lines["spread"]       # home always gets the opposite
        q_sp = query_spread(res, vis_line,
                            lines.get("vis_spread_odds"), lines.get("home_spread_odds"))
        _add("Spread", f"{vg.upper()} {vis_line:+.1f}",  q_sp["model_visitor_cover"], lines.get("vis_spread_odds"))
        _add("Spread", f"{hg.upper()} {home_line:+.1f}", q_sp["model_home_cover"],    lines.get("home_spread_odds"))

    # Total
    if "total" in markets and lines.get("total") is not None:
        q_tot = query_total(res, lines["total"],
                            lines.get("over_odds"), lines.get("under_odds"))
        _add("Total", f"Over  {lines['total']}", q_tot["model_over"],  lines.get("over_odds"))
        _add("Total", f"Under {lines['total']}", q_tot["model_under"], lines.get("under_odds"))

    return bets


GAME_IDS_CACHE_FILE = Path("game_xml_cache/team_game_ids.json")
MATRIX_CACHE_FILE   = Path("game_xml_cache/team_matrices.json")
GAME_IDS_TTL_HOURS  = 12   # re-scrape game IDs if older than this


def _load_game_ids_cache() -> dict:
    if GAME_IDS_CACHE_FILE.exists():
        return json.loads(GAME_IDS_CACHE_FILE.read_text())
    return {}


def _save_game_ids_cache(cache: dict) -> None:
    GAME_IDS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    GAME_IDS_CACHE_FILE.write_text(json.dumps(cache, indent=2))


def get_game_ids_cached(gid: str, n_games: int) -> list[str]:
    """Return game IDs for a team, using a TTL cache to avoid re-scraping."""
    import datetime as dt
    cache = _load_game_ids_cache()
    entry = cache.get(gid)

    if entry:
        age_hours = (dt.datetime.now().timestamp() - entry["ts"]) / 3600
        if age_hours < GAME_IDS_TTL_HOURS and len(entry["ids"]) >= n_games:
            log.info(f"[{gid}] Game IDs from cache (age {age_hours:.1f}h)")
            return entry["ids"][:n_games]

    # Cache miss or stale — scrape fresh
    ids = get_game_ids_for_team(gid, n_games)
    cache[gid] = {"ids": ids, "ts": dt.datetime.now().timestamp()}
    _save_game_ids_cache(cache)
    return ids


def _load_xmls(gid: str, n_games: int, xml_cache: Path) -> list[str]:
    """Fetch game IDs (with TTL cache) then download XMLs (file-cached forever)."""
    game_ids = get_game_ids_cached(gid, n_games)
    xmls = []
    for game_id in game_ids:
        try:
            xmls.append(fetch_game_xml(game_id, cache_dir=xml_cache))
        except Exception as e:
            log.warning(f"    Could not fetch XML {game_id}: {e}")
    return xmls


def run_screen(args) -> None:
    """Screen all games on a given date, rank all bet opportunities by edge."""
    from datetime import date as _date, datetime, timedelta
    from concurrent.futures import ThreadPoolExecutor, as_completed

    target_date = args.date or str(_date.today())
    log.info(f"Screening NCAAB games for {target_date} …")

    # ── Step 1: ESPN game list ────────────────────────────────────────────────
    espn_games = fetch_espn_games(target_date)
    if not espn_games:
        print(f"\nNo NCAAB games found on ESPN for {target_date}.")
        return

    # ── Step 2: Filter to specific games if --games provided ─────────────────
    if args.games:
        filters = [g.upper() for g in args.games]
        espn_games = [
            eg for eg in espn_games
            if any(f in eg["name"].upper() or
                   f in f"{eg['away_abbr']} VS {eg['home_abbr']}".upper()
                   for f in filters)
        ]
        log.info(f"Filtered to {len(espn_games)} games matching {filters}")
        if not espn_games:
            print(f"No games matched --games filter: {args.games}")
            return

    # ── Step 3: Odds API lines ────────────────────────────────────────────────
    odds_games = []
    if args.odds_api_key:
        date_obj  = datetime.strptime(target_date, "%Y-%m-%d")
        next_noon = (date_obj + timedelta(days=1)).strftime("%Y-%m-%dT12:00:00Z")
        params = {
            "apiKey":           args.odds_api_key,
            "regions":          "us",
            "markets":          "h2h,spreads,totals",
            "oddsFormat":       "american",
            "commenceTimeFrom": f"{target_date}T00:00:00Z",
            "commenceTimeTo":   next_noon,
        }
        resp = requests.get(ODDS_API_BASE, params=params, timeout=15)
        resp.raise_for_status()
        odds_games = resp.json()
        remaining = resp.headers.get("x-requests-remaining", "?")
        log.info(f"Odds API: {len(odds_games)} games with lines  (requests remaining: {remaining})")

    espn_games = match_odds_to_espn(espn_games, odds_games)

    # ── Step 4: Resolve GIDs ──────────────────────────────────────────────────
    known_gids  = load_or_scrape_gids()
    xml_cache   = Path(args.xml_dir) if args.xml_dir else Path("game_xml_cache")
    interactive = not args.no_interactive

    resolved = []
    skipped  = 0
    for eg in espn_games:
        away_abbr = eg.get("away_abbr", "").lower()
        home_abbr = eg.get("home_abbr", "").lower()
        vis_gid  = (odds_team_to_gid(away_abbr, known_gids, interactive=False)
                    or odds_team_to_gid(eg["away_team"], known_gids, interactive=interactive))
        home_gid = (odds_team_to_gid(home_abbr, known_gids, interactive=False)
                    or odds_team_to_gid(eg["home_team"], known_gids, interactive=interactive))
        if not vis_gid or not home_gid:
            log.warning(f"  Skipping {eg['away_team']} @ {eg['home_team']} — GID not resolved")
            skipped += 1
        else:
            resolved.append((eg, vis_gid, home_gid))

    # ── Step 5: Parallel simulation ───────────────────────────────────────────
    def process_game(eg, vis_gid, home_gid):
        lines = extract_lines(eg["odds"], preferred_book=args.book) if eg["odds"] else {}
        label = f"{eg['away_abbr'].upper()} @ {eg['home_abbr'].upper()}"
        bets  = simulate_game_for_screen(
            vis_gid, home_gid, lines, known_gids, xml_cache,
            n_games=args.n_games, weighting=args.weighting,
            nsim=args.nsim, kelly=args.kelly, markets=args.markets,
            game_label=label,
        )
        for b in bets:
            b["game_time"] = eg.get("time", "")
        return bets

    all_bets = []
    max_workers = min(6, len(resolved))
    log.info(f"Simulating {len(resolved)} games with {max_workers} parallel workers …")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_game, eg, vg, hg): eg
                   for eg, vg, hg in resolved}
        for fut in as_completed(futures):
            try:
                all_bets.extend(fut.result())
            except Exception as e:
                eg = futures[fut]
                log.warning(f"  Error on {eg['away_team']} @ {eg['home_team']}: {e}")

    if not all_bets:
        print(f"\nNo bets evaluated. ({skipped} games skipped)")
        return

    # ── Step 6: Sort and display ──────────────────────────────────────────────
    positive = [b for b in all_bets if b["edge"] > 0]

    if args.sort == "time":
        ranked = sorted(positive, key=lambda b: b.get("game_time", ""))
    else:
        ranked = sorted(positive, key=lambda b: b["edge"], reverse=True)

    SEP2 = "═" * 75
    print(f"\n{SEP2}")
    print(f"  DAILY SCREENER  —  {target_date}  [sorted by {args.sort}]")
    print(f"  {len(espn_games)} ESPN games  |  {len(odds_games)} with book lines  "
          f"|  {skipped} skipped  |  {len(positive)} positive-edge bets")
    print(f"  Showing top {min(args.top, len(ranked))}")
    print(SEP2)
    print(f"  {'#':<4} {'Game':<22} {'Time (UTC)':<13} {'Market':<9} {'Side':<22} "
          f"{'Model':>7} {'Book':>7} {'Edge':>7} {'Kelly':>8}")
    print(f"  {'─'*3} {'─'*21} {'─'*12} {'─'*8} {'─'*21} "
          f"{'─'*7} {'─'*7} {'─'*7} {'─'*8}")

    for i, b in enumerate(ranked[:args.top], 1):
        t = b.get("game_time", "")[:16].replace("T", " ") if b.get("game_time") else ""
        print(f"  {i:<4} {b['game']:<22} {t:<13} {b['market']:<9} {b['side']:<22} "
              f"{b['model_prob']*100:>6.1f}% {b['book_implied']*100:>6.1f}% "
              f"{b['edge']*100:>+6.1f}% {b['kelly']*100:>7.2f}%")

    print(SEP2)
    print(f"\n  Kelly % = fraction of bankroll to bet at {args.kelly}x Kelly\n")

def main() -> None:
    args = build_argparser().parse_args()

    xml_cache = Path(args.xml_dir) if args.xml_dir else Path("game_xml_cache")

    # ── Screen mode ──────────────────────────────────────────────────────────
    if args.screen:
        run_screen(args)
        return

    # ── Single game mode — require --visitor and --home ───────────────────────
    if not args.visitor or not args.home:
        raise SystemExit("ERROR: --visitor and --home are required (or use --screen)")

    # ── GID discovery ────────────────────────────────────────────────────────
    known_gids = load_or_scrape_gids(force_refresh=args.refresh_gids)

    if args.list_gids:
        print(f"\nKnown StatBroadcast team GIDs ({len(known_gids)} total):")
        for i, g in enumerate(known_gids):
            print(f"  {g}", end="\n" if (i+1) % 6 == 0 else "  ")
        print()
        return

    vis_gid  = resolve_gid(args.visitor, known_gids)
    home_gid = resolve_gid(args.home,    known_gids)
    log.info(f"Teams resolved — Visitor: '{vis_gid}'  Home: '{home_gid}'")

    # ── Scrape / load game XMLs ──────────────────────────────────────────────
    def load_xmls_for_team(gid: str) -> list[str]:
        log.info(f"=== Loading last {args.n_games} games for '{gid}' ===")
        game_ids = get_game_ids_for_team(gid, n_games=args.n_games)
        xmls = []
        for i, gid_id in enumerate(game_ids):
            xml_text = fetch_game_xml(gid_id, cache_dir=xml_cache)
            xmls.append(xml_text)
            if i < len(game_ids) - 1:
                time.sleep(args.rate_limit)
        return xmls

    vis_xmls  = load_xmls_for_team(vis_gid)
    time.sleep(args.rate_limit)
    home_xmls = load_xmls_for_team(home_gid)

    # ── Build blended matrices ───────────────────────────────────────────────
    log.info("Building blended transition matrices …")
    P_vis,  avg_poss_vis,  vis_name,  _         = blend_team_matrices(vis_xmls,  args.weighting)
    P_home, avg_poss_home, home_name_from_home,_ = blend_team_matrices(home_xmls, args.weighting)

    # Combine: visitor matrix carries visitor tendencies, home carries home
    P_match    = combine_visitor_home_matrices(P_vis, P_home)
    avg_poss   = int((avg_poss_vis + avg_poss_home) / 2)
    n_poss     = args.n_poss if args.n_poss else avg_poss

    # Always use the resolved GIDs as display names — the XML team names
    # reflect whoever was visitor/home in each historical game, not the
    # teams we're actually analysing.
    vis_display  = vis_gid.upper()
    home_display = home_gid.upper()

    # ── Auto-fetch lines from The Odds API ───────────────────────────────────
    if args.odds_api_key:
        log.info("Fetching lines from The Odds API …")
        try:
            games = fetch_odds(args.odds_api_key)
            game  = find_game(games, vis_gid, home_gid)
            if game:
                lines = extract_lines(game, preferred_book=args.book)
                print_fetched_lines(lines, vis_display, home_display)
                # Only fill in args that weren't explicitly provided
                if args.total         is None: args.total         = lines["total"]
                if args.over_odds     is None: args.over_odds     = lines["over_odds"]
                if args.under_odds    is None: args.under_odds    = lines["under_odds"]
                if args.spread        is None: args.spread        = lines["spread"]
                if args.vis_spread_odds  is None: args.vis_spread_odds  = lines["vis_spread_odds"]
                if args.home_spread_odds is None: args.home_spread_odds = lines["home_spread_odds"]
                if args.visitor_ml    is None: args.visitor_ml    = lines["visitor_ml"]
                if args.home_ml       is None: args.home_ml       = lines["home_ml"]
            else:
                log.warning("Game not found in odds feed — continuing without auto lines")
        except Exception as e:
            log.warning(f"Odds API fetch failed: {e} — continuing without auto lines")

    log.info(f"Visitor: {vis_display}  |  Home: {home_display}  |  n_possessions: {n_poss}")

    # ── Simulate ─────────────────────────────────────────────────────────────
    log.info(f"Simulating {args.nsim:,} games …")
    res = run_simulation(P_match, n_sim=args.nsim, n_possessions=n_poss, seed=args.seed)

    # ── Print simulation summary ──────────────────────────────────────────────
    print_simulation_summary(res, vis_display, home_display)

    # ── Betting queries ───────────────────────────────────────────────────────
    if args.total is not None:
        q = query_total(res, args.total, args.over_odds, args.under_odds, args.odds_format)
        print_total_query(q, kelly=args.kelly)

    if args.spread is not None:
        q = query_spread(res, args.spread,
                         args.vis_spread_odds, args.home_spread_odds, args.odds_format)
        print_spread_query(q, vis_display, home_display, kelly=args.kelly)

    if args.visitor_ml is not None or args.home_ml is not None:
        q = query_moneyline(res, args.visitor_ml, args.home_ml, args.odds_format)
        print_ml_query(q, vis_display, home_display, kelly=args.kelly)

    if args.query_variable and args.query_threshold is not None:
        q = query_custom(res, args.query_variable, args.query_direction, args.query_threshold)
        print_custom_query(q)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.plot:
        plot_results(res, vis_display, home_display, args.plot,
                     total_line=args.total, spread_line=args.spread)

    # ── JSON export ───────────────────────────────────────────────────────────
    if args.json:
        export = {k: v.tolist() if isinstance(v, np.ndarray) else v
                  for k, v in res.items()}
        Path(args.json).write_text(json.dumps(export, indent=2))
        log.info(f"Results JSON → '{args.json}'")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
