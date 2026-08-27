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
│  Output: predicted transition set         │
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

Everything in `src/goalpost/transitions/` stays as-is:
- `NFLTransitionModel` — extracts transition probabilities from a single game
- State discretization: down/distance buckets, field position buckets

### KalmanVAE (NEW — replaces 4-layer stack)

Three classes, total. That's it.

```python
class TransitionEncoder(nn.Module):
    """Encodes a game's transition probabilities into y_hat."""

    def __init__(self, input_dim: int, hidden_dim: int, y_dim: int):
        # input_dim = flattened transition matrix size
        # y_dim = observation vector size (what the Kalman sees)

    def forward(self, transition_probs: torch.Tensor) -> torch.Tensor:
        # Input: flat vector of P(next_state | state) for all states
        # Output: y_hat (observation encoding)
        pass


class KalmanFilter(nn.Module):
    """Updates team latent z_t using encoded observation y_hat."""

    def __init__(self, z_dim: int, y_dim: int):
        # z_dim = team latent dimension
        # y_dim = observation dimension (from encoder)

    def predict(self, z_prev: torch.Tensor) -> torch.Tensor:
        # State transition: z_t|t-1 = F * z_prev
        pass

    def update(self, z_pred: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        # Kalman gain + update: z_t|t = z_pred + K * (y_hat - H * z_pred)
        pass

    def forward(self, z_prev: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        # Convenience: predict then update
        pass


class TransitionDecoder(nn.Module):
    """Decodes a team latent into transition probabilities."""

    def __init__(self, z_dim: int, hidden_dim: int, output_dim: int):
        # z_dim = team latent dimension
        # output_dim = flattened transition matrix size (same as encoder input)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Input: team latent z
        # Output: reconstructed transition probabilities
        pass
```

### Inference (NEW — replaces GameModel + Simulator layering)

```python
class MatchupPredictor:
    """Mix two team latents and decode into a matchup transition set."""

    def __init__(self, decoder: TransitionDecoder, mixer: str = "average"):
        self.decoder = decoder
        self.mixer = mixer  # "average", "concat", or learned

    def predict(self, z_home: torch.Tensor, z_away: torch.Tensor) -> torch.Tensor:
        z_matchup = self._mix(z_home, z_away)
        return self.decoder(z_matchup)

    def _mix(self, z_home, z_away):
        if self.mixer == "average":
            return (z_home + z_away) / 2
        elif self.mixer == "concat":
            return torch.cat([z_home, z_away])
        # ... learned mixer is a small MLP
```

```python
class GameSimulator:
    """Simulate games using a predicted transition set."""

    def __init__(self, transition_model: NFLTransitionModel):
        # Takes a transition model (already populated with probabilities)
        # and runs Monte Carlo to get score distributions

    def simulate(self, n_games: int = 10000) -> List[GameOutcome]:
        pass
```

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
matchup_predictor = MatchupPredictor(decoder, mixer="average")
predicted_transitions = matchup_predictor.predict(z_home, z_away)

# Build a TransitionModel from the predicted probabilities
pred_model = NFLTransitionModel.from_probs(predicted_transitions)

# Simulate
sim = GameSimulator(pred_model)
outcomes = sim.simulate(n_games=10000)
```

## Why This Is Simpler

| Before (4 layers) | Now (3 classes) |
|---|---|
| DataSource → Representation → GameModel → Simulator | Data (keep) → KalmanVAE → Simulator |
| Representation learns z, GameModel learns z→transitions | Decoder learns z→transitions, Kalman updates z |
| Two separate learning problems | One end-to-end reconstruction loss |
| Need dataset of (z_home, z_away, transition_matrix) tuples | Just train encoder+kalman+decoder on single games |
| BayesianTeamUpdater is separate from VAE | Kalman subsumes Bayesian updating |

## Directory Structure

```
src/goalpost/
├── data/                    # KEEP — nflverse, ESPN, extraction
├── transitions/             # KEEP — NFLTransitionModel, state discretization
├── simulator/               # KEEP — MonteCarloSimulator (simplified)
├── kalman_vae/              # NEW — the entire model
│   ├── encoder.py           # TransitionEncoder
│   ├── kalman.py            # KalmanFilter
│   ├── decoder.py           # TransitionDecoder
│   ├── predictor.py         # MatchupPredictor
│   └── trainer.py           # Training loop
├── domain/                  # KEEP — Game, Possession, models
└── scripts/                 # KEEP + add kalman_vae training scripts
```

## Sport Expansion

| Sport | Data | Extractor | TransitionModel | KalmanVAE | Simulator |
|-------|------|-----------|-----------------|-----------|-----------|
| NFL | NFLVerseSource, ESPNSource | NFLDriveExtractor | NFLTransitionModel | Same encoder/kalman/decoder architecture | MonteCarloSimulator |
| NBA | NBAStatsSource | NBAPossessionExtractor | NBATransitionModel | Same | MonteCarloSimulator |
| MLB | MLBStatsSource | MLBInningExtractor | MLBTransitionModel | Same | MonteCarloSimulator |

Only the data layer and transition state space are sport-specific.
The KalmanVAE architecture is sport-agnostic — same encoder/kalman/decoder classes, different input dimensions.
