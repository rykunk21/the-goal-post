# GoalPost

NFL drive modeling. Kalman VAE for transition set generation, Monte Carlo simulation for outcomes.

## Architecture

Three moving parts:

1. **Data** — `NFLVerseSource` / `ESPNSource` / `UnifiedDataSource` + `NFLDriveExtractor`
2. **Kalman VAE** — `Encoder` → `KalmanFilter` → `Decoder`
3. **Simulation** — `MatchupPredictor` mixes two team latents, `MonteCarloSimulator` runs the game

## Project Structure

```
src/goalpost/
├── data/              # Data sources + drive extraction (KEEP)
│   ├── nflverse_source.py
│   ├── espn_source.py
│   ├── unified_source.py
│   └── nfl_drive_extractor.py
├── transitions/       # NFL transition model extraction (KEEP)
│   └── nfl_transition_model.py
├── simulator/         # Monte Carlo simulation (KEEP)
│   └── game_simulator.py
├── kalman_vae/        # NEW — the model
│   ├── encoder.py     # TransitionEncoder
│   ├── kalman.py      # KalmanFilter
│   ├── decoder.py     # TransitionDecoder
│   ├── predictor.py   # MatchupPredictor
│   └── trainer.py     # KalmanVAETrainer
└── domain/            # Game, Possession, Play models (KEEP)
```

## Quick Start

```python
from goalpost.kalman_vae import TransitionEncoder, KalmanFilter, TransitionDecoder, KalmanVAETrainer

encoder = TransitionEncoder(input_dim=512, y_dim=32)
kalman = KalmanFilter(z_dim=64, y_dim=32)
decoder = TransitionDecoder(z_dim=64, output_dim=512)

trainer = KalmanVAETrainer(encoder, kalman, decoder)

# Train on team game histories
team_games = {"KC": [game1, game2, ...], "BUF": [game1, game2, ...]}
losses = trainer.fit(team_games, n_epochs=10)

# Predict matchup
from goalpost.kalman_vae import MatchupPredictor
predictor = MatchupPredictor(decoder, mixer="average")
predicted = predictor(trainer.get_team_latent("KC"), trainer.get_team_latent("BUF"))
```

## Status

- Data pipeline: working (nflverse + ESPN)
- Transition extraction: working (play-level + drive-level)
- Kalman VAE: scaffolded, ready for training
- Simulator: working (Monte Carlo with transition models)
