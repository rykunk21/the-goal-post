# Kalman VAE

Three classes. One training loop. That's the whole model.

## Classes

| File | Class | Purpose |
|------|-------|---------|
| `encoder.py` | `TransitionEncoder` | Game transitions → y_hat (observation) |
| `kalman.py` | `KalmanFilterImpl` | z_prev + y_hat → z_new (updated team latent) |
| `decoder.py` | `TransitionDecoder` | z_new → reconstructed transitions |
| `predictor.py` | `MatchupPredictorImpl` | z_home + z_away → predicted matchup transitions |
| `trainer.py` | `KalmanVAETrainer` | Training loop + team latent history |

## Quick Start

```python
from goalpost.kalman_vae import TransitionEncoder, KalmanFilter, TransitionDecoder, KalmanVAETrainer

# Dimensions (example for NFL)
input_dim = 512   # flattened transition probabilities
y_dim = 32        # observation encoding
z_dim = 64        # team latent

encoder = TransitionEncoder(input_dim=input_dim, y_dim=y_dim)
kalman = KalmanFilter(z_dim=z_dim, y_dim=y_dim)
decoder = TransitionDecoder(z_dim=z_dim, output_dim=input_dim)

trainer = KalmanVAETrainer(encoder, kalman, decoder)

# Train
team_games = {
    "KC": [game1_tensor, game2_tensor, ...],   # each is [input_dim]
    "BUF": [game1_tensor, game2_tensor, ...],
}
losses = trainer.fit(team_games, n_epochs=10)

# Inference
from goalpost.kalman_vae import MatchupPredictor

predictor = MatchupPredictor(decoder, mixer="average")
z_home = trainer.get_team_latent("KC")
z_away = trainer.get_team_latent("BUF")

predicted = predictor(z_home, z_away)  # [input_dim] transition probabilities
```

## Key Design Decisions

1. **No separate VAE sampling**: We don't sample from a posterior distribution. The Kalman filter already handles uncertainty through its noise terms (Q, R). This makes training stable.

2. **z starts random**: A team's latent is initialized from N(0,1) and becomes meaningful after 2-3 games of Kalman updates. No pre-training needed.

3. **One loss**: Only reconstruction loss. The Kalman parameters (F, H, Q, R) learn implicitly from the reconstruction objective.

4. **Detachment between games**: After each game, z is detached from the computation graph. We don't backprop through an entire season — just through the current encoder→kalman→decoder step.

5. **Average mixer**: By default, matchup prediction averages the two team latents. This is surprisingly effective and requires no additional parameters. Swap to `concat` or `learned` if you need more expressiveness.
