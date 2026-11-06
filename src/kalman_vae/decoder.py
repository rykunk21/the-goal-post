"""Decoder: team latent → reconstructed transition probabilities."""

import torch
import torch.nn as nn

from ..abc.kalman_vae import Decoder


class TransitionDecoder(Decoder):
    """Decodes a team latent vector into transition probabilities.

    Input: z (team latent from Kalman filter)
    Output: reconstructed P(next_state | state) for all states

    This is the reconstruction head that makes the VAE trainable:
    the encoder produces y_hat, the Kalman updates z, and the decoder
    must reconstruct the original transition probabilities from z.
    """

    def __init__(
        self,
        z_dim: int = 64,
        hidden_dims: list[int] = [128, 256],
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = z_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

        self.output_dim = output_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode team latent into transition probabilities.

        Args:
            z: [batch, z_dim] or [z_dim]

        Returns:
            probs: [batch, output_dim] or [output_dim]
                Reconstructed transition probabilities (not yet softmaxed).
                Apply softmax per-state-group in the loss function.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        logits = self.net(z)

        if squeeze:
            logits = logits.squeeze(0)

        return logits
