# GoalPost — Three-Layer Architecture

```
┌─────────────────────────────────────────┐
│  Layer 1: DATA                          │
│  ├── DataSource (ABC)                   │
│  │   ├── NFLVerseSource                 │
│  │   ├── ESPNSource                     │
│  │   └── UnifiedDataSource              │
│  └── PossessionExtractor (ABC)         │
│      └── NFLDriveExtractor              │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Layer 2: REPRESENTATION              │
│  └── TeamRepresentation (ABC)           │
│      ├── BayesianTeamUpdater           │
│      └── VAEEncoder (future)            │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Layer 3: SIMULATION                    │
│  ├── TransitionModel (ABC)            │
│  │   └── NFLTransitionModel            │
│  └── Simulator (ABC)                    │
│      └── MonteCarloSimulator          │
└─────────────────────────────────────────┘
