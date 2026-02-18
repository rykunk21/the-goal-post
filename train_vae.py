#!/usr/bin/env python3
"""
Train VAE for NCAAB Prediction System.

Trains VAE to produce meaningful 16-dim latent representations from 80-dim team features.
Saves training curves and model checkpoints.

NOTE: This script requires REAL data from StatBroadcast. No synthetic data is used.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vae_infonce import VAE
from src.training.vae_trainer import (
    VAETrainer,
    prepare_training_data,
    extract_features_from_xml
)


def plot_training_curves(history: dict, output_path: str):
    """Plot and save training curves.
    
    Args:
        history: Training history dictionary
        output_path: Path to save plot
    """
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
    
    print(f"Saved training curves to {output_path}")


def save_history_to_csv(history: dict, output_path: str):
    """Save training history to CSV.
    
    Args:
        history: Training history dictionary
        output_path: Path to save CSV
    """
    df = pd.DataFrame(history)
    df.to_csv(output_path, index=False)
    print(f"Saved training history to {output_path}")


def analyze_latent_space(model: VAE, trainer: VAETrainer, xml_path: str):
    """Analyze latent space representations.
    
    Args:
        model: Trained VAE model
        trainer: VAE trainer
        xml_path: Path to sample game XML
    """
    print("\n" + "="*60)
    print("Latent Space Analysis")
    print("="*60)
    
    # Try to extract features from XML
    try:
        # Extract features for both teams
        msu_features = extract_features_from_xml(xml_path, 'MSU')
        ken_features = extract_features_from_xml(xml_path, 'KEN')
        
        # Encode to latent space
        msu_latent = trainer.encode(msu_features)
        ken_latent = trainer.encode(ken_features)
        
        print(f"\nMichigan State (MSU) - Winner (83-66):")
        print(f"  Feature vector (first 10): {msu_features[:10]}")
        print(f"  Latent mean: {msu_latent[0][:5]}...")
        print(f"  Latent std: {np.std(msu_latent[0]):.4f}")
        
        print(f"\nKentucky (KEN) - Loser (66-83):")
        print(f"  Feature vector (first 10): {ken_features[:10]}")
        print(f"  Latent mean: {ken_latent[0][:5]}...")
        print(f"  Latent std: {np.std(ken_latent[0]):.4f}")
        
        # Compute difference
        diff = np.abs(msu_latent - ken_latent)
        print(f"\nLatent space difference (absolute):")
        print(f"  Mean diff: {np.mean(diff):.4f}")
        print(f"  Max diff: {np.max(diff):.4f}")
        print(f"  Min diff: {np.min(diff):.4f}")
        
        # Check if latents are meaningful
        if np.mean(diff) > 0.1:
            print("\n✓ Latents are DIFFERENT - model encodes meaningful team differences")
        else:
            print("\n⚠ Latents are very similar - may indicate underfitting or collapse")
        
        return msu_latent, ken_latent
        
    except Exception as e:
        print(f"Could not analyze latent space: {e}")
        return None, None


def main():
    """Main training function.
    
    Loads game IDs from database, streams real XML data, and trains VAE.
    """
    print("="*60)
    print("VAE Training for NCAAB Prediction System")
    print("="*60)
    print("\nNOTE: Training requires REAL streaming data from StatBroadcast")
    print("No synthetic data fallback is available.\n")
    
    # Configuration
    INPUT_DIM = 80
    LATENT_DIM = 16
    LEARNING_RATE = 1e-3
    BETA = 1.0  # KL weight (balanced)
    EPOCHS = 200
    BATCH_SIZE = 32
    SEED = 42
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    MODEL_DIR = PROJECT_ROOT / "models" / "checkpoints"
    TRAINING_CURVES_DIR = PROJECT_ROOT / "training_curves"
    
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_CURVES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Step 1: Load game IDs from database
    print("\n[1/5] Loading game IDs from database...")
    try:
        from src.data.database import get_all_stored_game_ids, get_database_stats
        
        # Get available games
        all_games = get_all_stored_game_ids()
        stats = get_database_stats()
        
        print(f"  Total games in database: {stats['games_count']}")
        print(f"  Games with labels: {stats['games_with_labels']}")
        
        if stats['games_count'] == 0:
            raise ValueError(
                "No games found in database! "
                "Please run game discovery first to populate games."
            )
        
        game_ids = [g['game_id'] for g in all_games]
        print(f"  Loaded {len(game_ids)} game IDs")
        
    except Exception as e:
        print(f"Error loading from database: {e}")
        print("Falling back to streaming data discovery...")
        
        # Try to discover games from team schedules
        from src.data.team_gid_discovery import get_teams
        from src.data.game_discovery import discover_game_ids_from_teams
        
        teams = get_teams()
        team_gids = [t['gid'] for t in teams]
        print(f"  Found {len(team_gids)} teams")
        
        if not team_gids:
            raise ValueError(
                "No teams found! Cannot proceed with training. "
                "Please ensure team discovery has been run."
            )
        
        game_ids = discover_game_ids_from_teams(team_gids)
        unique_game_ids = list(set(game_ids))
        print(f"  Discovered {len(unique_game_ids)} unique game IDs")
        
        if len(unique_game_ids) == 0:
            raise ValueError(
                "No games discovered from team schedules! "
                "Cannot proceed with training."
            )
        
        game_ids = unique_game_ids
    
    # Step 2: Stream real XML features
    print("\n[2/5] Streaming game XML data...")
    try:
        from src.data.streaming_loader import StreamingXMLoader
        
        loader = StreamingXMLoader()
        all_features = []
        
        for i, game_id in enumerate(game_ids):
            try:
                home_feat, away_feat = loader.fetch_game_features(game_id)
                
                if home_feat is not None and away_feat is not None:
                    all_features.append(home_feat)
                    all_features.append(away_feat)
                    
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(game_ids)} games...")
                    
            except Exception as e:
                # Skip games that fail
                continue
        
        if not all_features:
            raise ValueError("No valid features extracted from any games!")
        
        print(f"  Collected {len(all_features)} team feature vectors")
        
    except Exception as e:
        print(f"Error streaming game data: {e}")
        raise ValueError(
            f"Failed to stream real game data: {e}. "
            "Cannot proceed without real data."
        )
    
    # Convert to numpy array
    features = np.array(all_features)
    print(f"  Feature matrix shape: {features.shape}")
    
    # Step 3: Prepare training data
    print("\n[3/5] Preparing training and validation data...")
    train_tensor, val_tensor = prepare_training_data(features, seed=SEED)
    
    print(f"  Training samples: {len(train_tensor)}")
    print(f"  Validation samples: {len(val_tensor)}")
    
    # Create data loaders
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Step 4: Initialize model and trainer
    print("\n[4/5] Initializing VAE model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    model = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, device=device)
    
    trainer = VAETrainer(
        model=model,
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        device=device,
        checkpoint_dir=str(MODEL_DIR)
    )
    
    print(f"  VAE Architecture:")
    print(f"    Encoder: {INPUT_DIM} → [128, 64] → {LATENT_DIM}")
    print(f"    Decoder: {LATENT_DIM} → [64, 128] → {INPUT_DIM}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 5: Train
    print(f"\n[5/5] Training VAE for {EPOCHS} epochs...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  KL weight (beta): {BETA}")
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        checkpoint_every=50,
        print_every=10,
        early_stopping_patience=30
    )
    
    # Save training curves
    print("\n[6/6] Saving training curves...")
    csv_path = TRAINING_CURVES_DIR / "vae_training.csv"
    save_history_to_csv(history, str(csv_path))
    
    plot_path = TRAINING_CURVES_DIR / "vae_loss.png"
    plot_training_curves(history, str(plot_path))
    
    # Save final model
    print("\n[7/7] Saving final model...")
    final_model_path = MODEL_DIR / "vae_trained.pt"
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': INPUT_DIM,
            'latent_dim': LATENT_DIM,
            'beta': BETA,
            'learning_rate': LEARNING_RATE
        },
        'history': history
    }, final_model_path)
    print(f"  Saved to: {final_model_path}")
    
    # Analyze latent space
    msu_latent, ken_latent = analyze_latent_space(model, trainer, XML_PATH)
    
    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    
    final_epoch = len(history['total_loss']) - 1
    print(f"  Final Epoch: {history['epoch'][final_epoch]}")
    print(f"  Total Loss: {history['total_loss'][final_epoch]:.4f}")
    print(f"  Reconstruction Loss: {history['recon_loss'][final_epoch]:.4f}")
    print(f"  KL Divergence: {history['kl_loss'][final_epoch]:.4f}")
    print(f"  Latent Std: {history['latent_std'][final_epoch]:.4f}")
    
    # Check for issues
    print("\n" + "-"*60)
    print("Analysis")
    print("-"*60)
    
    # Check KL collapse (KL too low means latents collapse to standard normal)
    final_kl = history['kl_loss'][final_epoch]
    if final_kl < 0.5:
        print(f"⚠️  WARNING: KL divergence is low ({final_kl:.4f})")
        print("    This may indicate latent collapse (latents becoming standard gaussian)")
    elif final_kl > 10:
        print(f"⚠️  WARNING: KL divergence is high ({final_kl:.4f})")
        print("    This may indicate KL dominance (reconstruction being ignored)")
    else:
        print(f"✓ KL divergence is healthy: {final_kl:.4f}")
    
    # Check reconstruction
    final_recon = history['recon_loss'][final_epoch]
    if final_recon < 0.2:
        print(f"✓ Reconstruction loss is good: {final_recon:.4f}")
    else:
        print(f"⚠️  Reconstruction loss is high: {final_recon:.4f}")
        print("    Model may not be capturing data well")
    
    # Check latent variance
    final_std = history['latent_std'][final_epoch]
    if final_std < 0.1:
        print(f"⚠️  WARNING: Latent std is low ({final_std:.4f})")
        print("    This indicates latent collapse")
    else:
        print(f"✓ Latent variance is meaningful: {final_std:.4f}")
    
    # Check training stability
    recent_losses = history['total_loss'][-10:]
    loss_variance = np.var(recent_losses)
    if loss_variance < 0.01:
        print(f"✓ Training is stable (variance in last 10 epochs: {loss_variance:.6f})")
    else:
        print(f"⚠️  Training may be unstable (variance: {loss_variance:.4f})")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    return model, trainer, history


if __name__ == "__main__":
    main()
