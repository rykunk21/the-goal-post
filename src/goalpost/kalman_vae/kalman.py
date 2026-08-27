"""Kalman filter for updating team latent vectors."""

import torch
import torch.nn as nn

from ..abc.kalman_vae import KalmanFilter


class KalmanFilterImpl(KalmanFilter):
    """Updates team latent z_t using encoded observation y_hat.

    Standard Kalman filter with learnable parameters:
    - F: state transition matrix
    - H: observation matrix
    - Q: process noise
    - R: observation noise
    - K: Kalman gain (computed, not learned directly)

    Forward pass:
        z_pred = F @ z_prev          (predict step)
        z_new  = z_pred + K @ (y_hat - H @ z_pred)   (update step)
    """

    def __init__(
        self,
        z_dim: int = 64,
        y_dim: int = 32,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.y_dim = y_dim

        # State transition: z_t = F * z_{t-1} + noise
        self.F = nn.Linear(z_dim, z_dim, bias=False)

        # Observation matrix: y = H * z + noise
        self.H = nn.Linear(z_dim, y_dim, bias=False)

        # Learnable log-variances for Q (process) and R (observation)
        self.log_Q = nn.Parameter(torch.zeros(z_dim))
        self.log_R = nn.Parameter(torch.zeros(y_dim))

        # Initialize F near identity (small updates per game)
        with torch.no_grad():
            self.F.weight.copy_(torch.eye(z_dim) * 0.95 + torch.randn(z_dim, z_dim) * 0.05)

    def predict(self, z_prev: torch.Tensor) -> torch.Tensor:
        """Predict step: z_t|t-1 = F * z_prev."""
        return self.F(z_prev)

    def update(self, z_pred: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Update step: incorporate observation y_hat.

        Computes Kalman gain K and updates:
            z_t|t = z_pred + K * (y_hat - H * z_pred)
        """
        # Predicted observation
        y_pred = self.H(z_pred)

        # Innovation (observation residual)
        innovation = y_hat - y_pred

        # Variances
        Q = torch.exp(self.log_Q)
        R = torch.exp(self.log_R)

        # Simplified Kalman gain (diagonal covariance assumption)
        # In full form: K = P_pred @ H.T @ (H @ P_pred @ H.T + R)^{-1}
        # Here we use a learned scalar weighting per dimension
        # This is the "deep Kalman filter" approximation
        H_sq = self.H.weight.pow(2).sum(dim=1)  # [y_dim]
        S = H_sq * Q.mean() + R  # simplified innovation covariance

        # Kalman gain (scalar per dimension for simplicity)
        K = (Q.mean() / S).unsqueeze(0)  # [1, y_dim]

        # Update: z_new = z_pred + K * innovation (broadcasted through H)
        # We map innovation back to z-space via H^T-like operation
        z_update = self.H.weight.T @ (K.squeeze(0) * innovation)
        z_new = z_pred + z_update

        return z_new

    def forward(self, z_prev: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Predict then update.

        Args:
            z_prev: [batch, z_dim] or [z_dim]
            y_hat: [batch, y_dim] or [y_dim]

        Returns:
            z_new: [batch, z_dim] or [z_dim]
        """
        if z_prev.dim() == 1:
            z_prev = z_prev.unsqueeze(0)
            y_hat = y_hat.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        z_pred = self.predict(z_prev)
        z_new = self.update(z_pred, y_hat)

        if squeeze:
            z_new = z_new.squeeze(0)

        return z_new
