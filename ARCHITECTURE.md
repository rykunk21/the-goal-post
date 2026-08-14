# GoalPost — High-Level Architecture

## Pipeline (5 stages)

```
Raw Data → DataSource → DriveExtractor → TeamRepresentation → TransitionModel → Simulator
```

## Abstract Contracts (Python ABC)

### 1. `DataSource` (ABC)
- `fetch(seasons: List[int]) → RawData` — pull from API/dataset
- `parse(raw: RawData) → List[Game]` — convert to domain objects
- `available_seasons() → List[int]` — what's cached/available

**Implementations:** `NFLVerseSource`, `SportradarSource`, `ESPNSource`, etc.

### 2. `DriveExtractor` (ABC)
- `extract(games: List[Game]) → List[Drive]` — group plays into drives
- `compute_state_transitions(drives: List[Drive]) → List[Transition]` — (state, action, next_state)

**Implementations:** `NFLDriveExtractor`, `NCAABDriveExtractor`, etc.

### 3. `TeamRepresentation` (ABC)
- `fit(drives: List[Drive]) → self` — learn from historical drives
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
    drives: List[Drive]

@dataclass
class Drive:
    drive_id: str
    team: str  # posteam
    plays: List[Play]
    result: DriveResult  # TD, FG, Punt, Turnover, End of Half/Game

@dataclass
class Play:
    play_id: str
    down: int
    distance: int
    yardline: int  # 1-100
    play_type: str  # pass, run, penalty, etc.
    yards_gained: int
    epa: float

@dataclass
class GameState:
    down: int
    distance: int
    yardline: int
    score_diff: int
    time_remaining: int  # seconds
    possession: str  # team with ball

@dataclass
class Transition:
    state: GameState
    action: str  # play category
    next_state: GameState
    team_id: str
    opponent_id: str

@dataclass
class GameOutcome:
    home_score: int
    away_score: int
    total: int
    margin: int
    first_half_margin: int  # for derivative markets
```

## Key Design Decisions

1. **Stage isolation** — Each ABC defines a clear output contract. Stage N only depends on stage N-1's output, not its implementation.
2. **Swappable sources** — `DataSource` handles NFL, college, etc. Same downstream code.
3. **Testable by stage** — Can validate `DriveExtractor` independently from `TeamRepresentation`.
4. **In-season updates** — `TeamRepresentation.update()` allows Bayesian/online updates without full retraining.

## Project Structure

```
goalpost/
├── src/
│   ├── goalpost/
│   │   ├── __init__.py
│   │   ├── abc/
│   │   │   ├── __init__.py
│   │   │   ├── data_source.py
│   │   │   ├── drive_extractor.py
│   │   │   ├── team_representation.py
│   │   │   ├── transition_model.py
│   │   │   └── simulator.py
│   │   ├── domain/
│   │   │   ├── __init__.py
│   │   │   └── models.py          # Game, Drive, Play, etc.
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── nflverse_source.py # DataSource impl
│   │   │   └── nfl_drive_extractor.py  # DriveExtractor impl
│   │   └── representation/
│   │       ├── __init__.py
│   │       └── vae_encoder.py     # TeamRepresentation impl
│   └── tests/
├── pyproject.toml
└── README.md
```

## Next Steps

1. Scaffold empty ABC files
2. Implement `NFLVerseSource` + `NFLDriveExtractor` with nflreadpy
3. Build a validation pipeline: download one season, extract drives, sanity-check counts
4. Then tackle representation learning on top of working data