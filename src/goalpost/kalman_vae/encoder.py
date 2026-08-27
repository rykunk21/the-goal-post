"""Encoder: transition probabilities → y_hat."""

import torch
import torch.nn as nn


class TransitionEncoder(nn.Module):
    """Encodes a game's transition probabilities into an observation vector.

    Input: flat vector of P(next_state | state) for all observed states
    Output: y_hat — encoded observation for the Kalman filter
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [256, 128],
        y_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, y_dim))
        self.net = nn.Sequential(*layers)

        self.y_dim = y_dim

    def forward(self, transition_probs: torch.Tensor) -> torch.Tensor:
        """Encode transition probabilities into y_hat.

        Args:
            transition_probs: [batch, input_dim] or [input_dim]
                Flattened transition probabilities from a single game.

        Returns:
            y_hat: [batch, y_dim] or [y_dim]
        """
        if transition_probs.dim() == 1:
            transition_probs = transition_probs.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        y_hat = self.net(transition_probs)

        if squeeze:
            y_hat = y_hat.squeeze(0)

        return y_hat
