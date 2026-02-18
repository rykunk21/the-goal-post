"""
VAE Trainer for NCAAB Prediction System.

Training pipeline for Variational Autoencoder to produce meaningful
16-dim latent representations from 80-dim team features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional, List
import numpy as np
from pathlib import Path
import json

from src.models.vae_infonce import VAE


class VAETrainer:
    """Trainer class for VAE model.
    
    Handles:
        - Data loading and batch processing
        - Training loop with loss tracking
        - Checkpoint saving
        - Learning curve logging
    """
    
    def __init__(
        self,
        model: Optional[VAE] = None,
        input_dim: int = 80,
        latent_dim: int = 16,
        learning_rate: float = 1e-3,
        beta: float = 1.0,  # KL weight
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "models/checkpoints"
    ):
        """Initialize VAE trainer.
        
        Args:
            model: VAE model (if None, creates new one)
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            learning_rate: Learning rate for optimizer
            beta: Weight for KL divergence term
            device: Device to use (CPU/GPU)
            checkpoint_dir: Directory to save checkpoints
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Create model if not provided
        if model is None:
            self.model = VAE(input_dim, latent_dim, self.device)
        else:
            self.model = model
            self.model.to(self.device)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5  # L2 regularization
        )
        
        # Learning rate scheduler - reduce on plateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-5
        )
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'epoch': [],
            'total_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'latent_std': []
        }
    
    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        """Compute VAE loss (reconstruction + KL).
        
        Args:
            x: Original input
            reconstruction: Reconstructed output
            mu: Latent mean
            logvar: Latent log variance
            
        Returns:
            Tuple of (total_loss, recon_loss, kl_loss)
        """
        # Reconstruction loss (binary cross entropy)
        recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='mean')
        
        # KL divergence
        kl_loss = self.model.kl_divergence(mu, logvar)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss.item(), kl_loss.item()
    
    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        """Single training step.
        
        Args:
            x: Input batch
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        reconstruction, mu, logvar = self.model(x)
        
        # Compute losses
        total_loss, recon_loss, kl_loss = self.compute_loss(x, reconstruction, mu, logvar)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Compute latent std for monitoring
        with torch.no_grad():
            z = self.model.reparameterize(mu, logvar)
            latent_std = z.std(dim=0).mean().item()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'latent_std': latent_std
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation data.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of average loss values
        """
        self.model.eval()
        
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_std = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                # Forward pass
                reconstruction, mu, logvar = self.model(x)
                
                # Compute losses
                loss, recon, kl = self.compute_loss(x, reconstruction, mu, logvar)
                
                # Latent std
                z = self.model.reparameterize(mu, logvar)
                latent_std = z.std(dim=0).mean().item()
                
                total_loss += loss.item()
                total_recon += recon
                total_kl += kl
                total_std += latent_std
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon / num_batches,
            'kl_loss': total_kl / num_batches,
            'latent_std': total_std / num_batches
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        checkpoint_every: int = 10,
        print_every: int = 10,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """Train VAE model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            checkpoint_every: Save checkpoint every N epochs
            print_every: Print progress every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            Training history dictionary
            
        Raises:
            NotImplementedError: If called without real streaming data (no synthetic fallback)
        """
        # Verify we have real data - raise error if no data loader
        if train_loader is None:
            raise NotImplementedError(
                "Training requires real streaming data. "
                "No synthetic data fallback available in production. "
                "Please provide train_loader with real game features."
            )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Training phase
            epoch_losses = {
                'total_loss': 0.0,
                'recon_loss': 0.0,
                'kl_loss': 0.0,
                'latent_std': 0.0
            }
            num_batches = 0
            
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                losses = self.train_step(x)
                
                for key in epoch_losses:
                    epoch_losses[key] += losses[key]
                num_batches += 1
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
            
            # Validation
            val_losses = None
            if val_loader is not None:
                val_losses = self.evaluate(val_loader)
            
            # Log to history
            self.history['epoch'].append(epoch)
            self.history['total_loss'].append(epoch_losses['total_loss'])
            self.history['recon_loss'].append(epoch_losses['recon_loss'])
            self.history['kl_loss'].append(epoch_losses['kl_loss'])
            self.history['latent_std'].append(epoch_losses['latent_std'])
            
            # Print progress
            if print_every and epoch % print_every == 0:
                msg = f"Epoch {epoch}/{epochs} | " \
                      f"Loss: {epoch_losses['total_loss']:.4f} | " \
                      f"Recon: {epoch_losses['recon_loss']:.4f} | " \
                      f"KL: {epoch_losses['kl_loss']:.4f} | " \
                      f"Latent Std: {epoch_losses['latent_std']:.4f}"
                
                if val_losses:
                    msg += f" | Val Loss: {val_losses['total_loss']:.4f}"
                
                print(msg)
            
            # Save checkpoint
            if checkpoint_every and epoch % checkpoint_every == 0:
                self.save_checkpoint(epoch, prefix="vae")
            
            # Step scheduler (reduce LR on plateau)
            if val_loader and val_losses:
                self.scheduler.step(val_losses['total_loss'])
            
            # Early stopping
            if early_stopping_patience and val_loader:
                if val_losses['total_loss'] < best_val_loss:
                    best_val_loss = val_losses['total_loss']
                    patience_counter = 0
                    # Save best model
                    self.save_checkpoint(epoch, prefix="best")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        return self.history
    
    def save_checkpoint(self, epoch: int, prefix: str = "vae") -> str:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            prefix: Filename prefix
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"{prefix}_epoch_{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'beta': self.beta
            }
        }, checkpoint_path)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        return checkpoint['epoch']
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent space.
        
        Args:
            x: Input features (numpy array)
            
        Returns:
            Latent representation (mu)
        """
        self.model.eval()
        
        # Convert to tensor
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        
        # Ensure correct shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            mu, logvar = self.model.encode(x)
        
        return mu.cpu().numpy()


def extract_features_from_xml(xml_path: str, team_id: str) -> np.ndarray:
    """Extract 80-dim team features from XML game data.
    
    Args:
        xml_path: Path to XML file
        team_id: Team identifier (e.g., 'MSU', 'KEN')
        
    Returns:
        80-dimensional feature vector
    """
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find the team element
    team_elem = None
    for team in root.findall('.//team'):
        if team.get('id') == team_id:
            team_elem = team
            break
    
    if team_elem is None:
        raise ValueError(f"Team {team_id} not found in XML")
    
    # Extract team totals
    totals = team_elem.find('.//totals/stats')
    special = team_elem.find('.//totals/special')
    
    # Initialize feature vector
    features = np.zeros(80)
    
    # Basic stats (0-13)
    features[0] = int(totals.get('tp', 0))  # Total points
    features[1] = float(totals.get('fgpct', 0)) / 100  # FG%
    features[2] = float(totals.get('fg3pct', 0)) / 100  # 3P%
    features[3] = float(totals.get('ftpct', 0)) / 100  # FT%
    features[4] = int(totals.get('fga', 0))  # FGA
    features[5] = int(totals.get('fga3', 0))  # 3PA
    features[6] = int(totals.get('fta', 0))  # FTA
    features[7] = int(totals.get('fgm', 0))  # FGM
    features[8] = int(totals.get('fgm3', 0))  # 3PM
    features[9] = int(totals.get('ftm', 0))  # FTM
    features[10] = int(totals.get('to', 0))  # Turnovers
    features[11] = int(totals.get('ast', 0))  # Assists
    features[12] = int(totals.get('blk', 0))  # Blocks
    features[13] = int(totals.get('stl', 0))  # Steals
    
    # Rebounds (14-19)
    features[14] = int(totals.get('treb', 0))  # Total rebounds
    features[15] = int(totals.get('oreb', 0))  # Offensive rebounds
    features[16] = int(totals.get('dreb', 0))  # Defensive rebounds
    features[17] = int(totals.get('pf', 0))  # Personal fouls
    features[18] = int(totals.get('min', 0))  # Minutes (usually 200)
    
    # Special stats (20-39)
    if special is not None:
        features[20] = int(special.get('pts_to', 0))  # Points off turnovers
        features[21] = int(special.get('pts_paint', 0))  # Paint points
        features[22] = int(special.get('pts_fastb', 0))  # Fast break points
        features[23] = int(special.get('pts_bench', 0))  # Bench points
        features[24] = int(special.get('pts_ch2', 0))  # Second chance points
        features[25] = int(special.get('poss_count', 0))  # Possessions
        features[26] = int(special.get('score_count', 0))  # Scoring possessions
        features[27] = int(special.get('ties', 0))  # Ties
        features[28] = int(special.get('leads', 0))  # Leads
        
        # Time-based features (convert from string format)
        lead_time = special.get('lead_time', '0')
        features[29] = int(lead_time) if lead_time.isdigit() else 0
    
    # The remaining features need opponent data - use zeros for now
    # or would need both teams' data
    
    return features


def prepare_training_data(
    features: np.ndarray,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare training and validation data from real features.
    
    Args:
        features: Real team features array (num_samples, 80)
        seed: Random seed
        
    Returns:
        Tuple of (train_tensor, val_tensor)
        
    Raises:
        ValueError: If no real features provided (no synthetic fallback)
    """
    # Require real data - no synthetic fallback
    if features is None or len(features) == 0:
        raise ValueError(
            "No real training data available. "
            "Training requires real streaming data from StatBroadcast. "
            "Cannot proceed with synthetic data in production."
        )
    
    print(f"Using {len(features)} real training samples")
    
    # Normalize features to [0, 1]
    # Use max of each feature for normalization (avoids hand-tuning ranges)
    feature_max = features.max(axis=0)
    feature_max = np.where(feature_max == 0, 1.0, feature_max)  # Avoid division by zero
    normalized = features / feature_max
    normalized = np.clip(normalized, 0, 1)
    
    # Split into train/val (80/20)
    np.random.seed(seed)
    indices = np.random.permutation(len(normalized))
    val_size = int(0.2 * len(normalized))
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_data = normalized[train_indices]
    val_data = normalized[val_indices]
    
    # Convert to tensors
    train_tensor = torch.FloatTensor(train_data)
    val_tensor = torch.FloatTensor(val_data)
    
    return train_tensor, val_tensor
