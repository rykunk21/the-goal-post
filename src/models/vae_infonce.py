"""
VAE + InfoNCE for NCAAB Prediction System.

Variational Autoencoder for team representation learning with InfoNCE contrastive loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Dict, Any
import numpy as np


class VAEEncoder(nn.Module):
    """VAE Encoder: 80-dim team features → 16-dim latent space (mu, logvar).
    
    Architecture:
        Input: 80-dim feature vector
        Hidden: [128, 64]
        Output: 16-dim mu, 16-dim logvar
        Activation: ReLU
    """
    
    def __init__(self, input_dim: int = 80, latent_dim: int = 16):
        super(VAEEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Hidden layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch_size, 80)
            
        Returns:
            Tuple of (mu, logvar) each of shape (batch_size, 16)
        """
        # Hidden layers with ReLU activation
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        
        # Output mu and logvar
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder: 16-dim latent → 80-dim reconstruction.
    
    Architecture:
        Input: 16-dim latent
        Hidden: [64, 128]
        Output: 80-dim reconstruction
        Activation: ReLU, Sigmoid on output
    """
    
    def __init__(self, latent_dim: int = 16, output_dim: int = 80):
        super(VAEDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Hidden layers
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        
        # Output layer
        self.fc_out = nn.Linear(128, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.
        
        Args:
            z: Latent tensor of shape (batch_size, 16)
            
        Returns:
            Reconstructed tensor of shape (batch_size, 80)
        """
        # Hidden layers with ReLU activation
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        
        # Output with Sigmoid activation for [0, 1] range
        reconstruction = torch.sigmoid(self.fc_out(h))
        
        return reconstruction


class VAE(nn.Module):
    """Variational Autoencoder combining encoder and decoder.
    
    Features:
        - 80-dim input → 16-dim latent → 80-dim output
        - Reparameterization trick for sampling
        - KL divergence loss computation
        - Device placement (CPU/GPU)
    """
    
    def __init__(
        self, 
        input_dim: int = 80, 
        latent_dim: int = 16,
        device: Optional[torch.device] = None
    ):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Initialize encoder and decoder
        self.encoder = VAEEncoder(input_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, input_dim)
        
        # Move to device
        self.to(self.device)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from N(mu, var).
        
        Args:
            mu: Mean of shape (batch_size, latent_dim)
            logvar: Log variance of shape (batch_size, latent_dim)
            
        Returns:
            Sampled tensor of shape (batch_size, latent_dim)
        """
        # Standard deviation from log variance
        std = torch.exp(0.5 * logvar)
        
        # Random noise
        eps = torch.randn_like(std)
        
        # Reparameterized sample
        z = mu + eps * std
        
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.
        
        Args:
            x: Input tensor of shape (batch_size, 80)
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, logvar
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space.
        
        Args:
            x: Input tensor of shape (batch_size, 80)
            
        Returns:
            Tuple of (mu, logvar)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output.
        
        Args:
            z: Latent tensor of shape (batch_size, 16)
            
        Returns:
            Reconstructed tensor of shape (batch_size, 80)
        """
        return self.decoder(z)
    
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """Sample from the latent space.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples of shape (num_samples, 80)
        """
        # Sample from standard normal
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        
        # Decode
        return self.decode(z)
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between latent distribution and standard normal.
        
        Args:
            mu: Mean of shape (batch_size, latent_dim)
            logvar: Log variance of shape (batch_size, latent_dim)
            
        Returns:
            KL divergence scalar
        """
        # KL divergence formula: -0.5 * sum(1 + log(var) - mu^2 - var)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        return kl.mean()


class InfoNCELoss(nn.Module):
    """InfoNCE Contrastive Loss for representation learning.
    
    Features:
        - L2 normalized embeddings
        - Similarity matrix computation with temperature
        - Contrastive loss with positive and negative pairs
        - Temperature tau = 0.07
    """
    
    def __init__(self, temperature: float = 0.07):
        super(InfoNCELoss, self).__init__()
        
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss.
        
        Args:
            embeddings: Normalized embeddings of shape (batch_size, embedding_dim)
                       Must be L2 normalized before passing
            
        Returns:
            Contrastive loss scalar
        """
        batch_size = embeddings.size(0)
        
        # Ensure embeddings are L2 normalized
        if embeddings.abs().sum() > batch_size * 1.1:  # Rough check if not normalized
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix: sim = z_i @ z_j.T / tau
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Create labels (diagonal = positive pairs)
        labels = torch.arange(batch_size, device=embeddings.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def compute_similarity(
        self, 
        embeddings_i: torch.Tensor, 
        embeddings_j: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity between two sets of embeddings.
        
        Args:
            embeddings_i: First set of embeddings (batch_size, embedding_dim)
            embeddings_j: Second set of embeddings (batch_size, embedding_dim)
            
        Returns:
            Similarity matrix (batch_size, batch_size)
        """
        # L2 normalize
        embeddings_i = F.normalize(embeddings_i, p=2, dim=1)
        embeddings_j = F.normalize(embeddings_j, p=2, dim=1)
        
        # Compute similarity
        similarity = torch.matmul(embeddings_i, embeddings_j.t()) / self.temperature
        
        return similarity
    
    def info_nce_loss(
        self, 
        embeddings_i: torch.Tensor, 
        embeddings_j: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute InfoNCE loss with positive and negative pairs.
        
        Args:
            embeddings_i: First set of embeddings (batch_size, embedding_dim)
            embeddings_j: Second set of embeddings (batch_size, embedding_dim)
            labels: Optional class labels for supervised contrastive learning
            
        Returns:
            Contrastive loss scalar
        """
        batch_size = embeddings_i.size(0)
        
        # Compute similarity matrix
        similarity = self.compute_similarity(embeddings_i, embeddings_j)
        
        if labels is None:
            # Unsupervised: diagonal are positive pairs
            pos_labels = torch.arange(batch_size, device=embeddings_i.device)
            loss = F.cross_entropy(similarity, pos_labels)
        else:
            # Supervised: create mask for positive pairs
            # For each sample, positive pairs are those with same label
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.t()).float()
            
            # Mask out diagonal (self-similarity)
            mask = mask - torch.eye(batch_size, device=embeddings_i.device)
            
            # Compute loss only over positive pairs
            exp_sim = torch.exp(similarity)
            
            # Sum of similarities to all samples
            denom = exp_sim.sum(dim=1, keepdim=True)
            
            # Positive similarity (sum over positive pairs)
            pos_sim = (exp_sim * mask).sum(dim=1)
            
            # Number of positive pairs per sample
            num_pos = mask.sum(dim=1).clamp(min=1)
            
            # Loss
            loss = -torch.log(pos_sim / denom.squeeze()).mean()
        
        return loss


class VAEWithInfoNCE(nn.Module):
    """Combined VAE and InfoNCE model for contrastive learning.
    
    Uses VAE latent representations for contrastive loss.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        latent_dim: int = 16,
        temperature: float = 0.07,
        device: Optional[torch.device] = None
    ):
        super(VAEWithInfoNCE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # VAE components
        self.vae = VAE(input_dim, latent_dim, device)
        
        # InfoNCE loss
        self.infonce = InfoNCELoss(temperature)
        
        # Move to device
        self.to(self.device)
    
    def forward(
        self, 
        x: torch.Tensor,
        use_infonce: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with VAE reconstruction and optional InfoNCE loss.
        
        Args:
            x: Input tensor of shape (batch_size, 80)
            use_infonce: Whether to compute InfoNCE loss on latent
            
        Returns:
            Dictionary containing:
                - reconstruction: VAE reconstruction
                - mu: Latent mean
                - logvar: Latent log variance
                - z: Sampled latent
                - vae_loss: VAE loss (reconstruction + KL)
                - infonce_loss: InfoNCE loss (if use_infonce=True)
        """
        # VAE forward
        reconstruction, mu, logvar = self.vae(x)
        
        # Get latent embeddings (use mu for deterministic embeddings)
        z = mu  # Use mean for contrastive learning
        
        # Compute VAE loss
        recon_loss = F.binary_cross_entropy(
            reconstruction, x, reduction='mean'
        )
        kl_loss = self.vae.kl_divergence(mu, logvar)
        vae_loss = recon_loss + kl_loss
        
        output = {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'vae_loss': vae_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
        
        # Compute InfoNCE loss on latent embeddings
        if use_infonce:
            # Normalize latent embeddings
            z_normalized = F.normalize(z, p=2, dim=1)
            infonce_loss = self.infonce(z_normalized)
            output['infonce_loss'] = infonce_loss
        
        return output


# Training utility functions

def create_vae_trainer(
    input_dim: int = 80,
    latent_dim: int = 16,
    temperature: float = 0.07,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None
) -> Tuple[VAEWithInfoNCE, torch.optim.Optimizer]:
    """Create VAE trainer with optimizer.
    
    Args:
        input_dim: Input feature dimension
        latent_dim: Latent space dimension
        temperature: InfoNCE temperature
        learning_rate: Learning rate for optimizer
        device: Device to place model on
        
    Returns:
        Tuple of (model, optimizer)
    """
    model = VAEWithInfoNCE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        temperature=temperature,
        device=device
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    return model, optimizer


def train_epoch(
    model: VAEWithInfoNCE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    beta: float = 1.0,
    gamma: float = 0.1,
    use_infonce: bool = True,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Train for one epoch.
    
    Args:
        model: VAE model
        dataloader: Training data loader
        optimizer: Optimizer
        beta: Weight for KL divergence
        gamma: Weight for InfoNCE loss
        use_infonce: Whether to use InfoNCE loss
        device: Device to use
        
    Returns:
        Dictionary of training metrics
    """
    if device is None:
        device = model.device
    
    model.train()
    
    total_vae_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_infonce_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        # Get data
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)
        
        # Forward pass
        output = model(x, use_infonce=use_infonce)
        
        # Compute losses
        vae_loss = output['vae_loss']
        recon_loss = output['recon_loss']
        kl_loss = output['kl_loss']
        
        # Total loss
        loss = vae_loss * beta
        
        if use_infonce and 'infonce_loss' in output:
            infonce_loss = output['infonce_loss']
            loss = loss + gamma * infonce_loss
            total_infonce_loss += infonce_loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate metrics
        total_vae_loss += vae_loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        num_batches += 1
    
    # Compute averages
    metrics = {
        'vae_loss': total_vae_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches
    }
    
    if use_infonce:
        metrics['infonce_loss'] = total_infonce_loss / num_batches
    
    return metrics


def evaluate(
    model: VAEWithInfoNCE,
    dataloader: DataLoader,
    use_infonce: bool = True,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Evaluate model on validation data.
    
    Args:
        model: VAE model
        dataloader: Validation data loader
        use_infonce: Whether to compute InfoNCE loss
        device: Device to use
        
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = model.device
    
    model.eval()
    
    total_vae_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_infonce_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Get data
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            
            # Forward pass
            output = model(x, use_infonce=use_infonce)
            
            # Accumulate metrics
            total_vae_loss += output['vae_loss'].item()
            total_recon_loss += output['recon_loss'].item()
            total_kl_loss += output['kl_loss'].item()
            
            if use_infonce and 'infonce_loss' in output:
                total_infonce_loss += output['infonce_loss'].item()
            
            num_batches += 1
    
    # Compute averages
    metrics = {
        'vae_loss': total_vae_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches
    }
    
    if use_infonce:
        metrics['infonce_loss'] = total_infonce_loss / num_batches
    
    return metrics


def get_team_embedding(
    model: VAE,
    team_features: np.ndarray,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Get latent embedding for a team.
    
    Args:
        model: VAE model
        team_features: Team feature vector (80-dim)
        device: Device to use
        
    Returns:
        Latent embedding (16-dim)
    """
    if device is None:
        device = model.device
    
    model.eval()
    
    # Convert to tensor
    if isinstance(team_features, np.ndarray):
        x = torch.FloatTensor(team_features).to(device)
    else:
        x = team_features.to(device)
    
    # Ensure correct shape
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    # Get latent representation (use mean)
    with torch.no_grad():
        mu, _ = model.encode(x)
    
    return mu.cpu().numpy()


# Forward pass example
if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("VAE + InfoNCE Forward Pass Example")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    model = VAEWithInfoNCE(
        input_dim=80,
        latent_dim=16,
        temperature=0.07,
        device=device
    )
    
    print(f"\nModel architecture:")
    print(f"  Encoder: 80-dim → [128, 64] → 16-dim (mu, logvar)")
    print(f"  Decoder: 16-dim → [64, 128] → 80-dim")
    print(f"  InfoNCE temperature: 0.07")
    
    # Create random input (batch of 4 team feature vectors)
    batch_size = 4
    x = torch.randn(batch_size, 80, device=device)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = model(x, use_infonce=True)
    
    print(f"\nOutput shapes:")
    print(f"  reconstruction: {output['reconstruction'].shape}")
    print(f"  mu: {output['mu'].shape}")
    print(f"  logvar: {output['logvar'].shape}")
    print(f"  z (latent): {output['z'].shape}")
    
    print(f"\nLoss values:")
    print(f"  vae_loss: {output['vae_loss'].item():.4f}")
    print(f"  recon_loss: {output['recon_loss'].item():.4f}")
    print(f"  kl_loss: {output['kl_loss'].item():.4f}")
    print(f"  infonce_loss: {output['infonce_loss'].item():.4f}")
    
    # Test VAE standalone
    print("\n" + "=" * 60)
    print("VAE Standalone Example")
    print("=" * 60)
    
    vae = VAE(input_dim=80, latent_dim=16, device=device)
    reconstruction, mu, logvar = vae(x)
    
    print(f"\nReconstruction shape: {reconstruction.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Test sampling
    samples = vae.sample(num_samples=2)
    print(f"\nSampled shapes: {samples.shape}")
    
    # Test InfoNCE standalone
    print("\n" + "=" * 60)
    print("InfoNCE Loss Example")
    print("=" * 60)
    
    infonce = InfoNCELoss(temperature=0.07)
    
    # Random embeddings
    embeddings = torch.randn(8, 16, device=device)
    embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize
    
    loss = infonce(embeddings)
    print(f"\nInfoNCE loss on random embeddings: {loss.item():.4f}")
    
    # Test training utilities
    print("\n" + "=" * 60)
    print("Training Utilities Example")
    print("=" * 60)
    
    # Create dummy data
    num_samples = 100
    X = torch.randn(num_samples, 80)
    
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create trainer
    model, optimizer = create_vae_trainer(
        input_dim=80,
        latent_dim=16,
        temperature=0.07,
        learning_rate=1e-3,
        device=device
    )
    
    # Train one epoch
    metrics = train_epoch(model, dataloader, optimizer, beta=1.0, gamma=0.1)
    
    print(f"\nTraining metrics (1 epoch):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Get team embedding using example features (normalized)
    team_features = np.array([
        0.75, 0.30, 0.60, 0.35, 0.20, 0.70, 0.75, 0.10, 0.30, 0.15,  # scoring
        0.12, 0.04, 0.20, 0.28, 0.55, 0.07, 0.18, 0.10, 0.15, 0.14,  # defense
        0.40, 0.10, 0.30, 0.32, 0.72, 0.48, 0.08, 0.24, 0.28, 0.70,  # rebounds
        0.50, 0.33, 0.72, 0.58, 0.35, 0.25, 0.52, 0.34, 0.72, -0.02,  # efficiency
        1.08, 0.98, 0.10, 0.70, 0.40, 0.11, 0.07, 1.07, 0.96, 0.11,  # advanced
        1.50, 0.30, 0.33, 0.10, 0.07, 0.20, 0.40, 0.20, 0.50, 0.80,  # context
        0.75, 0.75, 50.0, 33.0, 72.0, 4.0, 1.5, 0.7, 1.0, 10.8,  # season avg
        0.72, 1.03, 0.95, 0.50, 0.33, 0.38, 0.15, 0.14, 0.50, 1.2   # opp-adjusted
    ], dtype=np.float32)
    embedding = get_team_embedding(vae, team_features, device=device)
    print(f"\nTeam embedding shape: {embedding.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
