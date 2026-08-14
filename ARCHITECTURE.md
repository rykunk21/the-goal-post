# GoalPost — High-Level Architecture

## Pipeline (5 stages)

```
Raw Data → DataSource → PossessionExtractor → TeamRepresentation → TransitionModel → Simulator
```

Note: `PossessionExtractor` is the universal name. In NFL, a "possession" is called a "drive." The concept is the same: a contiguous sequence of plays where one team has the ball, ending in some terminal event (score, turnover, end of period, etc.).

## Abstract Contracts (Python ABC)

### 1. `DataSource` (ABC)
- `fetch(seasons: List[int]) → RawData` — pull from API/dataset
- `parse(raw: RawData) → List[Game]` — convert to domain objects
- `available_seasons() → List[int]` — what's cached/available

**Implementations:** `NFLVerseSource`, `SportradarSource`, `ESPNSource`, etc.

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

**Implementations:** `NFLDriveExtractor`, `NBAPossessionExtractor`, `MLBInningExtractor`, etc.

### 3. `TeamRepresentation` (ABC)
- `fit(possessions: List[Possession]) → self` — learn from historical possessions
- `encode(team_id: str, context: GameContext) → np.ndarray` — emit latent z
- `update(game: Game) → self` — in-season Bayesian update

**Implementations:** `VAEEncoder`, `ContrastiveEncoder`, `SlidingWindowEncoder`

### 4. `TransitionModel` (ABC)
- `fit(reps: Dict[str, np.ndarray], transitions: List[Transition]) → self`
- `predict(z_home: np.ndarray, z_away: np.ndarray, state: GameState) → TransitionMatrix`

**Implementations:** `MarkovTransitionModel`, `NeuralTransitionModel`

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
    # Universal
    play_type: str
    yards_gained: Optional[int] = None
    points_scored: int = 0
    epa: float = 0.0

@dataclass
class GameState:
    # NFL / CFB / continuous-field sports
    down: Optional[int] = None
    distance: Optional[int] = None
    yardline: Optional[int] = None
    # NBA / court sports
    shot_clock: Optional[int] = None
    # Universal
    score_diff: int
    time_remaining: int
    quarter: int = 1
    period: int = 1
    possession: str = ""
    timeouts_offense: int = 3
    timeouts_defense: int = 3

@dataclass
class Transition:
    state: GameState
    action: str
    next_state: GameState
    team_id: str
    opponent_id: str
    sport: str = "nfl"

@dataclass
class GameOutcome:
    home_score: int
    away_score: int
    total: int
    margin: int
    first_half_margin: int
    home_possessions: int = 0
    away_possessions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Key Design Decisions

1. **Stage isolation** — Each ABC defines a clear output contract. Stage N only depends on stage N-1's output, not its implementation.
2. **Swappable sources** — `DataSource` handles NFL, college, etc. Same downstream code.
3. **Testable by stage** — Can validate `PossessionExtractor` independently from `TeamRepresentation`.
4. **In-season updates** — `TeamRepresentation.update()` allows Bayesian/online updates without full retraining.
5. **Sport polymorphism** — Domain models use optional fields and sport tags rather than class hierarchies. A possession is a possession whether it's called a drive, an inning half, or a shift.

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
│   │   │   ├── nflverse_source.py
│   │   │   └── nfl_drive_extractor.py
│   │   └── representation/
│   │       ├── __init__.py
│   │       └── vae_encoder.py
│   └── tests/
├── pyproject.toml
└── README.md
```

## Next Steps

1. Implement `NFLVerseSource.parse()` and `NFLDriveExtractor` with nflreadpy
2. Build validation pipeline: download one season, extract possessions, sanity-check
3. Then tackle representation learning on top of working data