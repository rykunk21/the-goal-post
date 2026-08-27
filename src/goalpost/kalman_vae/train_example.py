"""Example: how to train the Kalman VAE encoder/decoder.

Pipeline:
    1. Load games from nflverse
    2. Extract transition matrices per team per game
    3. Flatten to fixed-size vectors
    4. Train encoder + kalman + decoder
    5. Save model
"""

import torch
from collections import defaultdict

from goalpost.data.nflverse_source import NFLVerseSource
from goalpost.transitions.nfl_transition_model import (
    NFLTransitionModel,
    extract_nfl_team_matrices,
)
from goalpost.kalman_vae import (
    TransitionEncoder,
    KalmanFilterImpl,
    TransitionDecoder,
    KalmanVAETrainer,
)


def build_team_games(seasons=[2023, 2024]) -> dict:
    """Load nflverse data and build {team_id: [game_tensors]}.

    Returns team_games ready for trainer.fit().
    """
    print("Fetching games from nflverse...")
    source = NFLVerseSource()
    source.fetch(seasons=seasons)  # Fetch data into internal cache
    games = source.parse()          # Parse cached data into Game objects
    print(f"Loaded {len(games)} games")

    # Group transition vectors by team, preserving game order
    team_games = defaultdict(list)

    for game in games:
        matrices = extract_nfl_team_matrices(game)
        for team_id, matrix in matrices.items():
            flat = matrix.flatten_probabilities()
            team_games[team_id].append({
                "tensor": torch.tensor(flat, dtype=torch.float32),
                "game_id": game.game_id,
                "date": getattr(game, "date", None),
            })

    # Sort each team's games chronologically by game metadata
    for team_id in team_games:
        team_games[team_id] = sorted(
            team_games[team_id],
            key=lambda g: (g["date"] or "", g["game_id"])
        )
        # Extract just the tensors for training
        team_games[team_id] = [g["tensor"] for g in team_games[team_id]]

    print(f"Built histories for {len(team_games)} teams")
    for team_id, games in sorted(team_games.items())[:5]:
        print(f"  {team_id}: {len(games)} games")

    return dict(team_games)


def main():
    # --- Build training data ---
    team_games = build_team_games(seasons=[2023, 2024])

    # --- Model dimensions ---
    input_dim = NFLTransitionModel.input_dim()  # 468
    y_dim = 32
    z_dim = 64

    print(f"\nModel dims: input={input_dim}, y={y_dim}, z={z_dim}")

    # --- Initialize components ---
    encoder = TransitionEncoder(input_dim=input_dim, y_dim=y_dim)
    kalman = KalmanFilterImpl(z_dim=z_dim, y_dim=y_dim)
    decoder = TransitionDecoder(z_dim=z_dim, output_dim=input_dim)

    # --- Train ---
    trainer = KalmanVAETrainer(encoder, kalman, decoder, lr=1e-3)

    print("\nTraining...")
    losses = trainer.fit(team_games, n_epochs=20)

    for epoch, loss in enumerate(losses):
        print(f"  Epoch {epoch+1}: loss={loss:.4f}")

    # --- Save ---
    trainer.save("kalman_vae_nfl.pt")
    print("\nSaved to kalman_vae_nfl.pt")

    # --- Inspect team latents ---
    print("\nTeam latents (sample):")
    for team_id in list(team_games.keys())[:5]:
        z = trainer.get_team_latent(team_id)
        print(f"  {team_id}: mean={z.mean():.3f}, std={z.std():.3f}")


if __name__ == "__main__":
    main()
