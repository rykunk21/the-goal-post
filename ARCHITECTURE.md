# GoalPost — High-Level Architecture

## Pipeline (5 stages)

```
Raw Data → DataSource → PossessionExtractor → TeamRepresentation → TransitionModel → Simulator
```

Note: `PossessionExtractor` is the universal name. In NFL, a "possession" is called a "drive." The concept is the same: a contiguous sequence of plays where one team has the ball, ending in some terminal event (score, turnover, end of period, etc.).

## Dual-Source Data Strategy

```
┌─────────────────────────────────────────────┐
│  UnifiedDataSource                            │
│  ├── NFLVerseSource (1999–2025)               │
│  │   → Rich: EPA, WPA, 372 columns           │
│  │   → Weekly batch updates                   │
│  │   → Canonical training data                │
│  └── ESPNSource (2026+, same-day)            │
│      → Immediate availability                │
│      → Basic features: yards, results, types   │
│      → Backfilled by nflverse when available   │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│  PossessionExtractor → State Transitions      │
│  └── Play-level + Drive-level transitions     │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│  BayesianTeamUpdater (Online)                 │
│  ├── Historical priors from nflverse fit      │
│  └── Live updates from ESPN games             │
│      → O(1) per drive, no retraining          │
│      → Vector: [TD%, FG%, Punt%, TO%, ...]   │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│  TransitionModel → Simulator                  │
│  └── Monte Carlo game simulation             │
└─────────────────────────────────────────────┘
```

### Why Two Sources?

- **nflverse** is the gold standard for historical data but lags 24-48 hours for new games
- **ESPN API** provides same-day data for live updating team representations
- **UnifiedDataSource** routes between them and tracks provenance for backfill

**Sunday workflow:** Train on nflverse → pull live ESPN games → update Bayesian posteriors → predict upcoming games with updated vectors → nflverse backfills Tuesday → full retrain.

## Abstract Contracts (Python ABC)

### 1. `DataSource` (ABC)
- `fetch(seasons: List[int]) → RawData` — pull from API/dataset
- `parse(raw: RawData) → List[Game]` — convert to domain objects
- `available_seasons() → List[int]` — what's cached/available

**Implementations:** `NFLVerseSource` (historical), `ESPNSource` (live), `UnifiedDataSource` (router)

### 2. `PossessionExtractor` (ABC)
- `extract(games: List[Game]) → List[Possession]` — group plays into possessions
- `compute_state_transitions(possessions: List[Possession]) → List[Transition]` — (state, action, next_state)

**Sport-specific mapping:**

| Sport | Possession Name | Terminal Events |
|-------|----------------|-----------------|
| NFL | Drive | TD, FG, Punt, Turnover, Safety, End of Half/Game |
| NBA | Possession | Made basket, turnover, shot clock violation, end of quarter |
| MLB | Inning half (or PA sequence) | 3 outs, runs scored, end of inning |
| NHL | Shift / Zone time | Goal, penalty, line change, end of period |
| Soccer | Possession | Goal, out of bounds, foul, end of half |
| CFB | Drive | Same as NFL |
| NCAAB | Possession | Made basket, turnover, shot clock, end of half |

**Implementations:** `NFLDriveExtractor`, `NCAABPossessionExtractor`, `MLBInningExtractor`

### 3. `TeamRepresentation` (ABC)
- `fit(possessions: List[Possession]) → self` — learn from historical possessions
- `encode(team_id: str, context: GameContext) → np.ndarray` — emit latent z
- `update(game: Game) → self` — in-season Bayesian update

**Implementations:**
- `BayesianTeamUpdater` — Beta-Bernoulli conjugate priors, O(1) online updates
- `VAEEncoder` — Variational autoencoder for deep representations (future)

### 4. `TransitionModel` (ABC)
- `fit(reps: Dict[str, np.ndarray], transitions: List[Transition]) → self`
- `predict(z_home: np.ndarray, z_away: np.ndarray, state: GameState) → TransitionMatrix`

**Implementations:** `MarkovTransitionModel` (empirical), `NeuralTransitionModel` (deep)

### 5. `Simulator` (ABC)
- `simulate(transition_model: TransitionModel, initial_state: GameState, n_sims: int) → List[GameOutcome]`
- `price(outcomes: List[GameOutcome], market: Market) → (prob, ev)`

**Implementations:** `MonteCarloSimulator`

## Domain Objects

```python
@dataclass
class Game:
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    possessions: List[Possession]
    home_score: int = 0
    away_score: int = 0
    sport: str = "nfl"

@dataclass
class Possession:
    """A contiguous sequence of plays where one team controls the ball.
    
    Called a 'drive' in NFL, 'possession' in NBA, 'inning half' in MLB, etc.
    """
    possession_id: str
    team: str
    plays: List[Play]
    result: PossessionResult
    quarter: int = 1
    start_field_position: Optional[int] = None
    end_field_position: Optional[int] = None
    points_scored: int = 0
    game_id: Optional[str] = None
    sport: str = "nfl"

class PossessionResult(Enum):
    """Possible end states of a possession. Sport-specific subclasses extend this."""
    # NFL / CFB
    TOUCHDOWN = "td"
    FIELD_GOAL = "fg"
    PUNT = "punt"
    TURNOVER = "turnover"
    TURNOVER_ON_DOWNS = "turnover_on_downs"
    SAFETY = "safety"
    # Universal
    END_OF_HALF = "end_of_half"
    END_OF_GAME = "end_of_game"
    # NBA
    MADE_BASKET = "made_basket"
    # MLB
    THREE_OUTS = "three_outs"
    # Hockey / Soccer
    GOAL = "goal"

@dataclass
class Play:
    play_id: str
    # NFL-specific (nullable for other sports)
    down: Optional[int] = None
    distance: Optional[int] = None
    yardline: Optional[int] = None
    # NBA / court sports
    shot_clock: Optional[int] = None
    # Universal
    play_type: str = ""
    yards_gained: Optional[int] = None
    points_scored: int = 0
    epa: float = 0.0
    wp: float = 0.5
    passer: Optional[str] = None
    rusher: Optional[str] = None
    receiver: Optional[str] = None
    penalty: bool = False
    turnover: bool = False
    scoring_play: bool = False

@dataclass
class GameState:
    # NFL / CFB / continuous-field sports
    down: Optional[int] = None
    distance: Optional[int] = None
    yardline: Optional[int] = None
    # NBA / court sports
    shot_clock: Optional[int] = None
    # Universal
    score_diff: int = 0
    time_remaining: int = 0
    quarter: int = 1
    period: int = 1
    possession: str = ""
    timeouts_offense: int = 3
    timeouts_defense: int = 3

@dataclass
class GameContext:
    """Context for encoding a team representation."""
    season: int = 0
    week: int = 0
    opponent_id: str = ""
    is_home: bool = False
    game_type: str = "REG"  # REG or POST

@dataclass
class Transition:
    """A state-action-next_state tuple for learning."""
    state: GameState = field(default_factory=GameState)
    action: str = ""  # play category or outcome bucket
    next_state: GameState = field(default_factory=GameState)
    team_id: str = ""
    opponent_id: str = ""
    possession_id: Optional[str] = None
    game_id: Optional[str] = None
    sport: str = "nfl"

@dataclass
class GameOutcome:
    """Result of a simulated game."""
    home_score: int = 0
    away_score: int = 0
    total: int = 0
    margin: int = 0
    first_half_margin: int = 0
    home_possessions: int = 0
    away_possessions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Market:
    """A betting market to price."""
    market_key: str = ""  # e.g. "spreads", "totals", "h2h"
    side: str = ""  # e.g. "home", "away", "over", "under"
    line: float = 0.0
    odds: float = 0.0  # American odds

# Type alias for transition matrix
TransitionMatrix = Dict[str, Dict[str, float]]  # from_state -> {to_state: prob}
```

## Key Design Decisions

1. **Stage isolation** — Each ABC defines a clear output contract. Stage N only depends on stage N-1's output, not its implementation.
2. **Dual-source routing** — `UnifiedDataSource` handles nflverse vs ESPN seamlessly, tracks provenance for backfill.
3. **Online updating** — `BayesianTeamUpdater` uses conjugate priors for O(1) per-drive updates without retraining.
4. **Swappable sources** — Same downstream code works regardless of data origin (nflverse or ESPN).
5. **Testable by stage** — Can validate `PossessionExtractor` independently from `TeamRepresentation`.
6. **Sport polymorphism** — Domain models use optional fields and sport tags rather than class hierarchies.

## Data Source Details

### NFL — nflverse / nflreadpy (Historical)

**Coverage:** 1999–present, all regular season + playoff games
**Format:** Polars DataFrame via `nflreadpy.load_pbp(seasons=[...])`
**Play fields:** play_id, game_id, drive, posteam, defteam, down, ydstogo, yardline_100, play_type, yards_gained, epa, wp, air_yards, passer, rusher, receiver, penalty, turnover, scoring_play
**Possession grouping:** Group by `(game_id, fixed_drive)` — sorted by `play_id` before grouping
**State transitions:** (down, distance, yardline, score_diff, time) → play outcome → next state
**Update cadence:** Weekly (Tuesdays after games complete)

### NFL — ESPN API (Live)

**Coverage:** Current season, same-day availability
**Format:** JSON via `site.api.espn.com`
**Play fields:** Text description, down, distance, yardLine, statYardage, scoringPlay, turnover
**Possession grouping:** `drives.previous` array, plays within each drive
**State transitions:** Same structure as nflverse but with basic features (no pre-computed EPA/WPA)
**Update cadence:** Real-time during games

### ESPN → Domain Mapping

| ESPN Field | Our Domain | Notes |
|---|---|---|
| `drives[].displayResult` | `PossessionResult` | Touchdown→TD, Field Goal→FG, Punt→PUNT, Interception→Turnover, Downs→Turnover_on_downs |
| `drives[].plays[].text` | `Play.play_type` | Parsed: pass, run, sack, punt, field_goal, etc. |
| `drives[].plays[].start.yardLine` | `Play.yardline` | 50=midfield, <50=own side |
| `drives[].plays[].statYardage` | `Play.yards_gained` | Net yards |
| `drives[].plays[].scoringPlay` | `Play.scoring_play` | Boolean |

### NCAAB — StatsBroadcast XML

**Endpoint:** Game-by-game XML feed
**Format:** XML with `<plays>` → `<play>` elements
**Play fields (XML → universal mapping):**

| StatsBroadcast XML | Universal `Play` field | Notes |
|---|---|---|
| `id` | `play_id` | |
| `type` | `play_type` | "2pt", "3pt", "ft", "rebound", "turnover", etc. |
| `team` | derived | which team has possession |
| `time` | derived | `GameState.time_remaining` |
| `score` | derived | `GameState.score_diff` |

**NCAAB `GameState` specifics:**
- `down`/`distance`/`yardline` → `None` (not applicable to basketball)
- `shot_clock` → populated if available (30 sec default)

## Bayesian Team Representation

### Vector Components (8-dimensional)

```
[0] P(TD)         — Posterior mean touchdown rate
[1] P(FG)         — Posterior mean field goal rate
[2] P(Punt)       — Posterior mean punt rate
[3] P(Turnover)   — Posterior mean turnover rate
[4] P(Turnover on downs) — Posterior mean turnover on downs rate
[5] P(Safety)     — Posterior mean safety rate
[6] Points/Drive  — Observed average
[7] Yards/Drive   — Observed average
```

### Update Rule (Beta-Bernoulli Conjugate Prior)

For each observed drive outcome:
- `alpha[outcome] += 1` (observe success)
- `beta[other_outcomes] += 1` (observe failure for others)
- Posterior mean = alpha / (alpha + beta)

Confidence grows with observation count: `conf = sqrt(n_drives) / sqrt(200)`

## Project Structure

```
goalpost/
├── src/
│   ├── goalpost/
│   │   ├── __init__.py
│   │   ├── abc/
│   │   │   ├── __init__.py
│   │   │   ├── data_source.py
│   │   │   ├── possession_extractor.py
│   │   │   ├── team_representation.py
│   │   │   ├── transition_model.py
│   │   │   └── simulator.py
│   │   ├── domain/
│   │   │   ├── __init__.py
│   │   │   └── models.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── nflverse_source.py      # Historical data (1999-2025)
│   │   │   ├── espn_source.py          # Live data (2026+)
│   │   │   ├── unified_source.py       # Router between sources
│   │   │   └── nfl_drive_extractor.py  # Drive extraction + transitions
│   │   ├── representation/
│   │   │   ├── __init__.py
│   │   │   ├── bayesian_updater.py     # Online Bayesian updates
│   │   │   └── vae_encoder.py          # Deep representations (future)
│   │   └── scripts/
│   │       ├── validate_pipeline.py    # End-to-end validation
│   │       └── test_espn_pipeline.py   # ESPN integration test
│   └── tests/
├── pyproject.toml
└── README.md
```

## Status

**Working:**
- ✅ nflverse ingestion (1999–2025)
- ✅ ESPN live ingestion (2026+ same-day)
- ✅ Unified data source with backfill tracking
- ✅ Domain model with correct drive results
- ✅ Play-level + drive-level transitions
- ✅ Bayesian online team updater
- ✅ Validation scripts

**Next:**
- ⏳ Discretize states for transition model
- ⏳ Build empirical transition matrix
- ⏳ Integrate Bayesian vectors into predictions
- ⏳ Monte Carlo simulator for market pricing
