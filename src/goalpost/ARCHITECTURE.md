# GoalPost — Kalman VAE Architecture

The simplest possible generative model for sports outcomes.

## Philosophy

- One model, not four layers
- Encoder + Kalman + Decoder = Transition Set
- Inference: mix two team latents → decode → simulate
- No separate "prediction layer" — the decoder IS the prediction

## Diagram

```
Training (per game, per team):
┌─────────────────────────────────────────┐
│  Transition Set from game data          │
│  (play transitions + drive results)     │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Encoder                                │
│  Input: transition probabilities        │
│  Output: y_hat (encoded observation)  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Kalman Update                          │
│  Input: y_hat, previous z_t             │
│  Output: z_{t+1} (updated team latent)  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Decoder (reconstruction loss)           │
│  Input: z_{t+1}                        │
│  Output: reconstructed transitions      │
│  Loss: ||reconstruction - input||       │
└─────────────────────────────────────────┘

Inference (upcoming game):
┌─────────────────────────────────────────┐
│  z_home, z_away (from Kalman updates)   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Mix Latents                            │
│  z_matchup = combine(z_home, z_away)    │
│  (concat, average, or learned)          │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Decoder                                │
│  Input: z_matchup                      │
│  Output: predicted transition set       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Simulator                              │
│  Run 10K games with predicted set       │
│  Output: score distribution             │
└─────────────────────────────────────────┘
```

## Components

### Data Pipeline (KEEP)

Everything in `src/goalpost/data/` stays as-is:
- `NFLVerseSource` / `ESPNSource` / `UnifiedDataSource`
- `NFLDriveExtractor` — play-by-play → possessions → transitions

### Transition Model (KEEP, sport-specific)

Everything in `src/goalpost/transitions/` stays as-is:
- `NFLTransitionModel` — extracts transition probabilities from a single game
- State discretization: down/distance buckets, field position buckets
- **New:** `flatten_probabilities()` and `from_flat_probabilities()` for encoder/decoder I/O

### Abstract Base Classes (NEW — sport-agnostic contracts)

In `src/goalpost/abc/`:

| ABC | File | Role |
|---|---|---|
| `DataSource` | `data_source.py` | Fetch + parse raw data → Game objects |
| `PossessionExtractor` | `possession_extractor.py` | Plays → possessions → transitions |
| `TransitionModel` | `transition_model.py` | Extract + simulate transitions per team per game |
| `Simulator` | `simulator.py` | Monte Carlo from transition model → outcomes |
| `Encoder` | `kalman_vae.py` | transition_probs → y_hat |
| `KalmanFilter` | `kalman_vae.py` | z_prev + y_hat → z_new |
| `Decoder` | `kalman_vae.py` | z → reconstructed transition_probs |
| `MatchupPredictor` | `kalman_vae.py` | z_home + z_away → predicted transitions |
| `Trainer` | `kalman_vae.py` | Training loop + latent history |

### Kalman VAE Implementations (NEW — current, NFL-specific)

In `src/goalpost/kalman_vae/`:

| Class | Inherits | File | Role |
|---|---|---|---|
| `TransitionEncoder` | `Encoder` | `encoder.py` | Game transitions → y_hat |
| `KalmanFilterImpl` | `KalmanFilter` | `kalman.py` | z_prev + y_hat → z_new |
| `TransitionDecoder` | `Decoder` | `decoder.py` | z → reconstructed transitions |
| `MatchupPredictorImpl` | `MatchupPredictor` | `predictor.py` | z_home + z_away → predicted transitions |
| `KalmanVAETrainer` | `Trainer` | `trainer.py` | Training loop + latent history |

Sport-specific implementations will subclass the ABCs in `abc.kalman_vae` with different dimensions, architectures, or state spaces.

## Directory Structure

```
src/goalpost/
├── abc/                   # Abstract base classes (sport-agnostic)
│   ├── data_source.py
│   ├── possession_extractor.py
│   ├── transition_model.py
│   ├── simulator.py
│   └── kalman_vae.py      # Encoder, KalmanFilter, Decoder, MatchupPredictor, Trainer
├── data/                  # KEEP — nflverse, ESPN, extraction
├── transitions/           # KEEP — NFLTransitionModel (sport-specific)
├── simulator/             # KEEP — MonteCarloSimulator
├── kalman_vae/            # NEW — NFL-specific Kalman VAE implementations
│   ├── encoder.py
│   ├── kalman.py
│   ├── decoder.py
│   ├── predictor.py
│   └── trainer.py
├── domain/                # KEEP — Game, Possession, Play models
└── scripts/               # KEEP + add kalman_vae training scripts
```

## Sport Expansion

| Sport | DataSource | PossessionExtractor | TransitionModel | KalmanVAE | Simulator |
|-------|------------|---------------------|-----------------|-----------|-----------|
| NFL | NFLVerseSource, ESPNSource | NFLDriveExtractor | NFLTransitionModel | TransitionEncoder, KalmanFilterImpl, TransitionDecoder | MonteCarloSimulator |
| NBA | NBAStatsSource | NBAPossessionExtractor | NBATransitionModel | Same ABCs, different dims | MonteCarloSimulator |
| MLB | MLBStatsSource | MLBInningExtractor | MLBTransitionModel | Same ABCs, different dims | MonteCarloSimulator |

Only the data layer, transition state space, and encoder/decoder dimensions are sport-specific.
The Kalman filter architecture and training loop are sport-agnostic.

## Training Loop

```python
# For each team, maintain z_t across their game history
z = torch.randn(z_dim)  # Initialize randomly

for game in team_games:
    # Extract transition probabilities from this game
    transition_probs = extract_transition_probs(game, team_id)

    # Encode → Kalman update → decode
    y_hat = encoder(transition_probs)
    z = kalman(z, y_hat)  # predict + update
    reconstructed = decoder(z)

    # Loss: reconstruct the input transitions
    loss = mse(reconstructed, transition_probs)
    loss.backward()
    optimizer.step()
```

Key: z starts random, becomes a proper team latent after a few games.
No separate "representation learning" phase — it emerges from the reconstruction objective.

## Inference Loop

```python
# Before a game: both teams have z from their recent games
z_home = kalman_history["KC"]  # updated through Week 5
z_away = kalman_history["BUF"]  # updated through Week 5

# Mix and decode
matchup_predictor = MatchupPredictorImpl(decoder, mixer="average")
predicted_transitions = matchup_predictor.predict(z_home, z_away)

# Build a TransitionModel from the predicted probabilities
pred_model = NFLTransitionModel.from_flat_probabilities(predicted_transitions, "KC", "game_123", "BUF")

# Simulate
sim = GameSimulator(pred_model)
outcomes = sim.simulate(n_sims=10000)
```
