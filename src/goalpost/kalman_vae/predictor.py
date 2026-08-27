"""Matchup predictor: mix two team latents and decode."""

import torch
import torch.nn as nn


class MatchupPredictor(nn.Module):
    """Mix two team latents and decode into a matchup transition set.

    At inference time:
        z_home = kalman_history["KC"]   # after Week 5 games
        z_away = kalman_history["BUF"]  # after Week 5 games
        z_matchup = mix(z_home, z_away)
        predicted_transitions = decoder(z_matchup)
    """

    def __init__(
        self,
        decoder: nn.Module,
        z_dim: int = 64,
        mixer: str = "average",
    ):
        super().__init__()

        self.decoder = decoder
        self.z_dim = z_dim
        self.mixer = mixer

        if mixer == "concat":
            # Learned mixer: project 2*z_dim back to z_dim
            self.mixer_net = nn.Linear(z_dim * 2, z_dim)
        elif mixer == "learned":
            # Full learned combination (e.g., bilinear or MLP)
            self.mixer_net = nn.Sequential(
                nn.Linear(z_dim * 2, z_dim),
                nn.ReLU(),
                nn.Linear(z_dim, z_dim),
            )
        elif mixer == "average":
            self.mixer_net = None
        else:
            raise ValueError(f"Unknown mixer: {mixer}")

    def _mix(self, z_home: torch.Tensor, z_away: torch.Tensor) -> torch.Tensor:
        """Combine two team latents into a matchup latent."""
        if self.mixer == "average":
            return (z_home + z_away) / 2

        # Concatenate and project
        z_cat = torch.cat([z_home, z_away], dim=-1)
        return self.mixer_net(z_cat)

    def forward(
        self,
        z_home: torch.Tensor,
        z_away: torch.Tensor,
    ) -> torch.Tensor:
        """Predict matchup transition probabilities.

        Args:
            z_home: [batch, z_dim] or [z_dim]
            z_away: [batch, z_dim] or [z_dim]

        Returns:
            predicted_transitions: [batch, output_dim] or [output_dim]
        """
        if z_home.dim() == 1:
            z_home = z_home.unsqueeze(0)
            z_away = z_away.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        z_matchup = self._mix(z_home, z_away)
        predicted = self.decoder(z_matchup)

        if squeeze:
            predicted = predicted.squeeze(0)

        return predicted
