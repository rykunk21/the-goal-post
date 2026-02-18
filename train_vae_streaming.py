#!/usr/bin/env python3
"""
Train VAE with Streaming Data from StatBroadcast.

Trains VAE on real NCAAB game data streamed from StatBroadcast archive.
Uses 80-dim features extracted from XML, trains to produce 16-dim latent space.

Key features:
- Streams games from StatBroadcast (rate limited 1/sec)
- 80/20 train/test split with probability
- Caches test set for reproducible evaluation
- Logs progress every 100 batches
- Saves checkpoints every epoch
"""

import torch
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import json
import logging
from datetime import datetime
from typing import Optional, List, Tuple

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vae_infonce import VAE
from src.training.vae_trainer import VAETrainer
from src.data.streaming_loader import StreamingXMLoader, get_game_ids_for_streaming, discover_game_ids_from_teams
from src.data.team_gid_discovery import get_teams
from src.training.streaming_dataset import StreamingTeamDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_training_curves(history: dict, output_path: str):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = history['epoch']
    
    # Total Loss
    axes[0, 0].plot(epochs, history['total_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss (Reconstruction + KL)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction Loss
    axes[0, 1].plot(epochs, history['recon_loss'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss (BCE)')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL Divergence
    axes[1, 0].plot(epochs, history['kl_loss'], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence (Latent Regularization)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Latent Standard Deviation
    axes[1, 1].plot(epochs, history['latent_std'], 'm-', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='k', linestyle='--', label='Standard Normal')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Latent Std (mean across dims)')
    axes[1, 1].set_title('Latent Space Variance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training curves to {output_path}")


def save_history_to_csv(history: dict, output_path: str):
    """Save training history to CSV."""
    df = pd.DataFrame(history)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved training history to {output_path}")


def stream_games_to_samples(
    game_ids: List[int],
    loader: StreamingXMLoader,
    max_samples: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Stream games and collect feature samples.
    
    Args:
        game_ids: List of game IDs to stream
        loader: Streaming XML loader
        max_samples: Maximum samples to collect
        
    Returns:
        Tuple of (home_features_list, away_features_list)
        
    Raises:
        ValueError: If no valid games found (no synthetic fallback)
    """
    home_samples = []
    away_samples = []
    
    for game_id in game_ids:
        if max_samples and len(home_samples) >= max_samples:
            break
            
        try:
            home_feat, away_feat = loader.fetch_game_features(game_id)
            
            if home_feat is not None and away_feat is not None:
                home_samples.append(home_feat)
                away_samples.append(away_feat)
                
                if len(home_samples) % 50 == 0:
                    logger.info(f"Streamed {len(home_samples)} games...")
                    
        except Exception as e:
            # Log but continue - some games may fail
            logger.debug(f"Skipping game {game_id}: {e}")
            continue
    
    if not home_samples:
        raise ValueError(
            "No valid games found from streaming data! "
            "Cannot proceed with training - no synthetic data fallback available."
        )
    
    return home_samples, away_samples


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to [0, 1] range.
    
    Args:
        features: Raw feature array
        
    Returns:
        Normalized features
    """
    feature_max = features.max(axis=0)
    feature_max = np.where(feature_max == 0, 1.0, feature_max)
    normalized = features / feature_max
    return np.clip(normalized, 0, 1)


def train_vae_streaming(
    game_ids: List[int],
    input_dim: int = 80,
    latent_dim: int = 16,
    learning_rate: float = 1e-3,
    beta: float = 1.0,
    epochs: int = 50,
    batch_size: int = 32,
    test_prob: float = 0.2,
    checkpoint_dir: str = "models/checkpoints",
    cache_dir: str = "data/test_cache",
    log_every: int = 100,
    max_games: Optional[int] = None
) -> Tuple[VAE, VAETrainer, dict]:
    """Train VAE with streaming game data.
    
    Args:
        game_ids: Game IDs to stream
        input_dim: Input feature dimension (80)
        latent_dim: Latent space dimension (16)
        learning_rate: Learning rate
        beta: KL divergence weight
        epochs: Number of epochs
        batch_size: Batch size
        test_prob: Probability for test split
        checkpoint_dir: Directory for checkpoints
        cache_dir: Directory for test cache
        log_every: Log every N batches
        max_games: Maximum games to process
        
    Returns:
        Tuple of (trained_model, trainer, history)
    """
    # Setup paths
    PROJECT_ROOT = Path(__file__).parent.parent
    checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    cache_dir = PROJECT_ROOT / cache_dir
    training_curves_dir = PROJECT_ROOT / "training_curves"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    training_curves_dir.mkdir(parents=True, exist_ok=True)
    
    # Limit games if specified
    if max_games:
        game_ids = game_ids[:max_games]
    
    logger.info(f"Starting streaming VAE training with {len(game_ids)} games")
    
    # Initialize streaming loader
    loader = StreamingXMLoader()
    
    # Stream games and collect samples
    logger.info("Streaming game data from StatBroadcast...")
    home_samples, away_samples = stream_games_to_samples(game_ids, loader, max_games)
    
    if not home_samples:
        raise ValueError("No valid games found!")
    
    # Combine home and away samples
    all_features = np.array(home_samples + away_samples)
    logger.info(f"Collected {len(all_features)} total samples")
    
    # Normalize features
    all_features = normalize_features(all_features)
    
    # Split into train and test
    np.random.seed(42)
    indices = np.random.permutation(len(all_features))
    test_size = int(len(all_features) * test_prob)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    train_features = all_features[train_indices]
    test_features = all_features[test_indices]
    
    # Cache test set with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_cache_path = cache_dir / f"test_samples_{timestamp}.npz"
    np.savez(
        test_cache_path,
        features=test_features,
        timestamp=timestamp,
        num_samples=len(test_features)
    )
    logger.info(f"Cached {len(test_features)} test samples to {test_cache_path}")
    
    # Convert to tensors
    train_tensor = torch.FloatTensor(train_features)
    test_tensor = torch.FloatTensor(test_features)
    
    logger.info(f"Training samples: {len(train_tensor)}")
    logger.info(f"Test samples: {len(test_tensor)}")
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = VAE(input_dim=input_dim, latent_dim=latent_dim, device=device)
    
    trainer = VAETrainer(
        model=model,
        input_dim=input_dim,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        beta=beta,
        device=device,
        checkpoint_dir=str(checkpoint_dir)
    )
    
    logger.info(f"VAE Architecture: {input_dim} → [128, 64] → {latent_dim}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop with logging
    logger.info(f"Training for {epochs} epochs...")
    
    history = {
        'epoch': [],
        'total_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'latent_std': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        epoch_losses = {'total': 0, 'recon': 0, 'kl': 0, 'std': 0}
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            x = batch[0].to(device)
            
            # Forward pass
            trainer.optimizer.zero_grad()
            reconstruction, mu, logvar = model(x)
            
            # Compute losses
            recon_loss = torch.nn.functional.binary_cross_entropy(reconstruction, x)
            kl_loss = model.kl_divergence(mu, logvar)
            total_loss = recon_loss + beta * kl_loss
            
            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            trainer.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                z = model.reparameterize(mu, logvar)
                latent_std = z.std(dim=0).mean().item()
            
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['kl'] += kl_loss.item()
            epoch_losses['std'] += latent_std
            num_batches += 1
            
            # Log every N batches
            if log_every and batch_idx % log_every == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Recon: {recon_loss.item():.4f} | "
                    f"KL: {kl_loss.item():.4f}"
                )
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Validation
        model.eval()
        val_losses = {'total': 0, 'recon': 0, 'kl': 0, 'std': 0}
        val_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(device)
                reconstruction, mu, logvar = model(x)
                
                recon_loss = torch.nn.functional.binary_cross_entropy(reconstruction, x)
                kl_loss = model.kl_divergence(mu, logvar)
                total_loss = recon_loss + beta * kl_loss
                
                z = model.reparameterize(mu, logvar)
                latent_std = z.std(dim=0).mean().item()
                
                val_losses['total'] += total_loss.item()
                val_losses['recon'] += recon_loss.item()
                val_losses['kl'] += kl_loss.item()
                val_losses['std'] += latent_std
                val_batches += 1
        
        for key in val_losses:
            val_losses[key] /= val_batches
        
        # Log epoch summary
        logger.info(
            f"Epoch {epoch}/{epochs} Summary: "
            f"Train Loss: {epoch_losses['total']:.4f} | "
            f"Val Loss: {val_losses['total']:.4f} | "
            f"Recon: {epoch_losses['recon']:.4f} | "
            f"KL: {epoch_losses['kl']:.4f} | "
            f"Latent Std: {epoch_losses['std']:.4f}"
        )
        
        # Save to history
        history['epoch'].append(epoch)
        history['total_loss'].append(epoch_losses['total'])
        history['recon_loss'].append(epoch_losses['recon'])
        history['kl_loss'].append(epoch_losses['kl'])
        history['latent_std'].append(epoch_losses['std'])
        
        # Save checkpoint every epoch
        checkpoint_path = checkpoint_dir / f"vae_streaming_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'history': history,
            'config': {
                'input_dim': input_dim,
                'latent_dim': latent_dim,
                'beta': beta,
                'learning_rate': learning_rate,
                'num_games': len(game_ids),
                'timestamp': timestamp
            }
        }, checkpoint_path)
        
        # Track best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_checkpoint = checkpoint_dir / "vae_streaming_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': best_val_loss
            }, best_checkpoint)
    
    # Save final model
    final_model_path = checkpoint_dir / "vae_streaming.pt"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {
            'input_dim': input_dim,
            'latent_dim': latent_dim,
            'beta': beta,
            'learning_rate': learning_rate,
            'num_games': len(game_ids),
            'timestamp': timestamp
        }
    }, final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Save training curves
    save_history_to_csv(history, str(training_curves_dir / "vae_streaming.csv"))
    plot_training_curves(history, str(training_curves_dir / "vae_streaming_loss.png"))
    
    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Games streamed: {len(game_ids)}")
    print(f"Training samples: {len(train_tensor)}")
    print(f"Test samples: {len(test_tensor)}")
    print(f"Epochs: {epochs}")
    print(f"Final train loss: {history['total_loss'][-1]:.4f}")
    print(f"Final val loss: {val_losses['total']:.4f}")
    print(f"Final recon loss: {history['recon_loss'][-1]:.4f}")
    print(f"Final KL loss: {history['kl_loss'][-1]:.4f}")
    print(f"Final latent std: {history['latent_std'][-1]:.4f}")
    print("="*60)
    
    return model, trainer, history


def main():
    """Main training function.
    
    NOTE: This script requires REAL data from StatBroadcast. No synthetic data is used.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train VAE with streaming game data')
    parser.add_argument('--games', type=int, default=1000, help='Number of games to stream')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0, help='KL weight')
    parser.add_argument('--max_games', type=int, default=None, help='Max games to process')
    parser.add_argument('--log_every', type=int, default=100, help='Log every N batches')
    parser.add_argument('--no_cache', action='store_true', help='Force re-discovery of game IDs')
    parser.add_argument('--use_cached', action='store_true', help='Skip discovery, use cached game IDs only')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("VAE Streaming Training for NCAAB Prediction System")
    print("="*60)
    print("NOTE: Training requires REAL streaming data from StatBroadcast")
    print("No synthetic data fallback is available.\n")
    
    # Discover game IDs using proper GID -> Schedule -> Game ID flow
    logger.info("Discovering game IDs from team schedules...")
    
    # Get teams (from cache or discover)
    teams = get_teams()
    team_gids = [t['gid'] for t in teams]
    logger.info(f"Using {len(team_gids)} teams for game discovery")
    
    if not team_gids:
        raise ValueError(
            "No teams found! Cannot proceed with training. "
            "Please run team discovery first to populate team data."
        )
    
    # Use cached game IDs if requested or if discovery fails
    unique_game_ids = []
    
    if args.use_cached:
        logger.info("Using cached game IDs (--use_cached flag)")
        cache_file = Path(__file__).parent / 'data' / 'statbroadcast_game_ids.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            for team, ids in cached.items():
                unique_game_ids.extend(ids)
            unique_game_ids = list(set(unique_game_ids))
            logger.info(f"Loaded {len(unique_game_ids)} cached game IDs")
        else:
            raise ValueError("No cached game IDs found!")
    else:
        # Discover game IDs from team schedules
        game_ids = discover_game_ids_from_teams(team_gids)
        unique_game_ids = list(set(game_ids))
        logger.info(f"Found {len(game_ids)} total games ({len(unique_game_ids)} unique)")
        
        # Fallback to cached game IDs if no games discovered
        if len(unique_game_ids) == 0:
            logger.warning("No games discovered from schedules, falling back to cached game IDs")
            cache_file = Path(__file__).parent / 'data' / 'statbroadcast_game_ids.json'
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                for team, ids in cached.items():
                    unique_game_ids.extend(ids)
                unique_game_ids = list(set(unique_game_ids))
                logger.info(f"Loaded {len(unique_game_ids)} cached game IDs")
            else:
                raise ValueError(
                    "No games discovered from team schedules! "
                    "Cannot proceed with training - no synthetic data fallback available."
                )
    
    # Log discovery summary - requested format: "Found X teams, Y total games (Z unique)"
    logger.info(f"Found {len(team_gids)} teams, {len(unique_game_ids)} total games ({len(unique_game_ids)} unique)")
    
    # Limit to requested number if specified
    if args.max_games:
        unique_game_ids = unique_game_ids[:args.max_games]
    
    # Train
    model, trainer, history = train_vae_streaming(
        game_ids=unique_game_ids,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=args.beta,
        log_every=args.log_every,
        max_games=args.max_games
    )
    
    return model, trainer, history


if __name__ == "__main__":
    main()
