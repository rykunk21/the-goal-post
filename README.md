# Basketball Betting Simulator

MCMC-based NCAAB betting simulator. Scrapes play-by-play data from StatBroadcast,
builds transition matrices, simulates 50k games, and compares model probabilities
against book lines to find edges.

## Setup

```bash
# Python dependencies
pip install requests numpy matplotlib

# Node dependencies (for game ID scraping)
npm install puppeteer

# Get a free Odds API key (500 req/month free)
# https://the-odds-api.com

# Set your API key in your environment
export ODDS_API=your_key_here
```

## Files

| File | Purpose |
|------|---------|
| `basketball_betting.py` | Main script |
| `get_game_ids.js` | Puppeteer scraper for StatBroadcast game IDs |
| `game_xml_cache/` | Auto-created cache dir for XMLs, GIDs, name mappings |

---

## Commands

### Single Game

Analyze one specific matchup with auto-fetched lines:

```bash
python basketball_betting.py \
  --visitor okst \
  --home hou \
  --odds-api-key $SPORT_API \
  --kelly 0.25 \
  --plot output.png
```

With a specific book:

```bash
python basketball_betting.py \
  --visitor okst \
  --home hou \
  --odds-api-key $SPORT_API \
  --book draftkings \
  --kelly 0.25 \
  --plot output.png
```

With manual lines (no API key needed):

```bash
python basketball_betting.py \
  --visitor okst \
  --home hou \
  --total 130.5 --over-odds -110 --under-odds -110 \
  --spread -7.5 --vis-spread-odds -110 --home-spread-odds -110 \
  --visitor-ml +260 --home-ml -320 \
  --kelly 0.25 \
  --plot output.png
```

Filter to specific markets only:

```bash
python basketball_betting.py \
  --visitor okst --home hou \
  --odds-api-key $SPORT_API \
  --book draftkings \
  --kelly 0.25 \
  --markets spread total
```

---

### Daily Screener

Screen all games today, ranked by edge:

```bash
python basketball_betting.py \
  --screen \
  --odds-api-key $SPORT_API \
  --kelly 0.25
```

With a specific book:

```bash
python basketball_betting.py \
  --screen \
  --odds-api-key $SPORT_API \
  --book draftkings \
  --kelly 0.25
```

Filter to spreads only:

```bash
python basketball_betting.py \
  --screen \
  --odds-api-key $SPORT_API \
  --book draftkings \
  --kelly 0.25 \
  --markets spread
```

Screen a specific date:

```bash
python basketball_betting.py \
  --screen \
  --date 2026-03-07 \
  --odds-api-key $SPORT_API \
  --book fanduel \
  --kelly 0.25 \
  --top 15
```

Unattended (skip uncertain team matches instead of prompting):

```bash
python basketball_betting.py \
  --screen \
  --odds-api-key $SPORT_API \
  --kelly 0.25 \
  --no-interactive
```

---

## All Flags

### Teams
| Flag | Description |
|------|-------------|
| `--visitor GID` | Visitor team StatBroadcast GID (e.g. `okst`) |
| `--home GID` | Home team StatBroadcast GID (e.g. `hou`) |

### Odds / Lines
| Flag | Description |
|------|-------------|
| `--odds-api-key KEY` | The Odds API key for auto-fetching lines |
| `--book BOOK` | Preferred book: `draftkings`, `fanduel`, `betmgm`, `caesars` |
| `--total LINE` | Over/under line, e.g. `130.5` |
| `--over-odds ODDS` | American odds for over, e.g. `-110` |
| `--under-odds ODDS` | American odds for under, e.g. `-110` |
| `--spread LINE` | Spread from visitor perspective, e.g. `-4.5` |
| `--vis-spread-odds ODDS` | Visitor spread odds |
| `--home-spread-odds ODDS` | Home spread odds |
| `--visitor-ml ODDS` | Visitor moneyline |
| `--home-ml ODDS` | Home moneyline |

### Screener
| Flag | Description |
|------|-------------|
| `--screen` | Run daily screener mode |
| `--date YYYY-MM-DD` | Date to screen (default: today) |
| `--top N` | Number of top bets to show (default: 10) |
| `--markets` | Filter markets: `ml`, `spread`, `total` (default: all) |
| `--no-interactive` | Skip uncertain GID matches instead of prompting |

### Simulation
| Flag | Description |
|------|-------------|
| `--nsim N` | Number of simulations (default: 50000) |
| `--n-games N` | Recent games to blend per team (default: 5) |
| `--n-poss N` | Override possessions per game |
| `--weighting` | `recency` (default) or `equal` |
| `--kelly FLOAT` | Kelly fraction, e.g. `0.25` = quarter Kelly (default: `1.0`) |
| `--seed N` | Random seed (default: 42) |

### Output
| Flag | Description |
|------|-------------|
| `--plot FILE` | Save simulation plot to PNG |
| `--json FILE` | Save full results to JSON |
| `--list-gids` | Print all known StatBroadcast team GIDs |
| `--refresh-gids` | Re-scrape the GID list even if cached |

---

## Kelly Criterion

Kelly stake = `(b×p - q) / b × fraction` where:
- `p` = model win probability
- `q` = 1 - p
- `b` = net decimal odds

Common fractions:
- `--kelly 1.0` — full Kelly (aggressive, mathematically optimal)
- `--kelly 0.5` — half Kelly (recommended for most bettors)
- `--kelly 0.25` — quarter Kelly (conservative, lower variance)

## GID Cache

The first time you run the screener, uncertain team name matches will prompt
you to select the correct StatBroadcast GID. Confirmed mappings are saved to
`game_xml_cache/name_to_gid.json` and reused automatically on future runs.

To see all available GIDs:
```bash
python basketball_betting.py --list-gids
```
