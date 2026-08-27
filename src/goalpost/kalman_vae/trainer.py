"""Training loop for Kalman VAE."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

from .encoder import TransitionEncoder
from .kalman import KalmanFilter
from .decoder import TransitionDecoder


class KalmanVAETrainer:
    """Train encoder + kalman + decoder on team game histories.

    For each team:
        z = random init
        for game in chronological order:
            transitions = extract_from_game(game)
            y_hat = encoder(transitions)
            z = kalman(z, y_hat)
            reconstructed = decoder(z)
            loss = mse(reconstructed, transitions)
    """

    def __init__(
        self,
        encoder: TransitionEncoder,
        kalman: KalmanFilter,
        decoder: TransitionDecoder,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.encoder = encoder.to(device)
        self.kalman = kalman.to(device)
        self.decoder = decoder.to(device)
        self.device = device

        self.optimizer = optim.Adam(
            list(encoder.parameters())
            + list(kalman.parameters())
            + list(decoder.parameters()),
            lr=lr,
        )

        self.criterion = nn.MSELoss()

        # Team latent history: {team_id: z_tensor}
        self.team_latents: Dict[str, torch.Tensor] = {}

    def fit(
        self,
        team_games: Dict[str, List[torch.Tensor]],
        n_epochs: int = 10,
    ) -> List[float]:
        """Train on team game histories.

        Args:
            team_games: {team_id: [game_transition_tensors]}
                Each tensor is a flat vector of transition probabilities.

        Returns:
            epoch_losses: list of average loss per epoch
        """
        epoch_losses = []

        for epoch in range(n_epochs):
            total_loss = 0.0
            n_updates = 0

            for team_id, games in team_games.items():
                # Initialize or retrieve team latent
                if team_id not in self.team_latents:
                    z = torch.randn(self.kalman.z_dim, device=self.device)
                else:
                    z = self.team_latents[team_id].clone()

                for transitions in games:
                    transitions = transitions.to(self.device)

                    self.optimizer.zero_grad()

                    # Forward pass
                    y_hat = self.encoder(transitions)
                    z = self.kalman(z, y_hat)
                    reconstructed = self.decoder(z)

                    # Loss: reconstruct the input transitions
                    loss = self.criterion(reconstructed, transitions)

                    loss.backward()
                    self.optimizer.step()

                    # Detach z for next game (no backprop through entire history)
                    z = z.detach()

                    total_loss += loss.item()
                    n_updates += 1

                # Store final z for this team
                self.team_latents[team_id] = z.detach()

            avg_loss = total_loss / max(n_updates, 1)
            epoch_losses.append(avg_loss)

        return epoch_losses

    def get_team_latent(self, team_id: str) -> torch.Tensor:
        """Get the current latent for a team."""
        if team_id not in self.team_latents:
            return torch.randn(self.kalman.z_dim, device=self.device)
        return self.team_latents[team_id]

    def save(self, path: str):
        """Save model state."""
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "kalman": self.kalman.state_dict(),
                "decoder": self.decoder.state_dict(),
                "team_latents": self.team_latents,
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.kalman.load_state_dict(checkpoint["kalman"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.team_latents = checkpoint["team_latents"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
