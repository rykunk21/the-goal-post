# ESPN Live Integration — Implementation Summary

## What Was Built

### 1. `ESPNSource` (`data/espn_source.py`)
Fetches live/recent NFL games from ESPN's public API.

**Features:**
- `fetch_game(game_id)` — Pull specific game by ESPN event ID
- `fetch_date_range(start, end)` — Pull all games in a date range
- `load_game(game_id)` — Convenience: fetch + parse in one call
- Full domain model conversion (ESPN JSON → Game/Possession/Play)

**ESPN → Domain Mapping:**
| ESPN Field | Our Domain | Notes |
|---|---|---|
| `drives[].displayResult` | `PossessionResult` | TD, FG, Punt, Interception→Turnover, Downs→Turnover_on_downs |
| `drives[].plays[].text` | `Play.play_type` | Parsed for pass/run/sack/punt/field_goal/etc |
| `drives[].plays[].start.yardLine` | `Play.yardline` | Own yard line (50=midfield) |
| `drives[].plays[].statYardage` | `Play.yards_gained` | Net yards |
| `drives[].plays[].scoringPlay` | `Play.scoring_play` | Boolean |

**Verified Working:** Lions @ Bengals Aug 13 2026 — 24 drives, 14-16 final score.

### 2. `BayesianTeamUpdater` (`representation/bayesian_updater.py`)
Online team representation updater using Beta-Bernoulli conjugate priors.

**How it works:**
1. **Initialize** with historical data → sets Beta priors from observed rates
2. **Observe new game** → increment alpha (success) or beta (failure) for each outcome
3. **Posterior mean** becomes the updated rate estimate
4. **Vector** = [P(TD), P(FG), P(Punt), P(Turnover), P(Turnover_on_downs), P(Safety), Points/Drive, Yards/Drive]

**Key feature:** Updates in O(1) per drive — no retraining needed.

**Verified Working:** After adding the Lions @ Bengals game:
- DET TD rate dropped from 30.0% → 19.2% (preseason offenses struggle)
- CIN TD rate dropped from 25.4% → 16.9%

### 3. `UnifiedDataSource` (`data/unified_source.py`)
Routes between nflverse and ESPN intelligently.

**Usage:**
```python
from goalpost.data import UnifiedDataSource

source = UnifiedDataSource()

# Load historical + live
games = source.load(
    seasons=[2023, 2024, 2025],
    live_game_ids=["401873272"]  # Lions @ Bengals Aug 13
)

# Check source
game = source.get_game("401873272")
source_name = source.get_source_for_game("401873272")  # "espn"
```

## Architecture

```
┌─────────────────────────────────────────────┐
│  UnifiedDataSource                          │
│  ├── NFLVerseSource (1999–2025)             │
│  │   → Rich: EPA, WPA, 372 cols             │
│  │   → Weekly updates                       │
│  └── ESPNSource (2026+, same-day)           │
│      → Immediate availability                │
│      → Basic: yards, results, play types     │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│  BayesianTeamUpdater                        │
│  ├── Historical priors (nflverse fit)       │
│  └── Live updates (ESPN games)              │
│      → O(1) per drive                      │
│      → Backfill when nflverse available     │
└─────────────────────────────────────────────┘
```

## Workflow for Sunday Games

1. **Saturday night:** Model trained on nflverse 1999–2025
2. **Sunday 1 PM:** Kickoff
3. **Sunday 4:30 PM:** `espn.fetch_live_date_range(today, today)` pulls all games
4. **Sunday 4:31 PM:** `updater.update(game)` for each completed game
5. **Sunday 4:32 PM:** Updated team vectors ready for 4:25 PM or SNF predictions
6. **Tuesday:** nflverse releases Week X data, backfill replaces ESPN versions
7. **Tuesday night:** Full retrain on complete dataset

## Files Added/Modified

- `data/espn_source.py` — NEW: ESPN API client
- `data/unified_source.py` — NEW: Router between sources
- `representation/bayesian_updater.py` — NEW: Online Bayesian updater
- `data/__init__.py` — Updated exports
- `representation/__init__.py` — Updated exports
- `scripts/test_espn_pipeline.py` — NEW: Validation script

## Next Steps

1. **Discretize states** for transition model (yardline bins, time bins)
2. **Build empirical transition matrix** from drive-level transitions
3. **Integrate Bayesian vectors** into transition probability predictions
4. **Add simulator** that uses updated vectors for next-week predictions
