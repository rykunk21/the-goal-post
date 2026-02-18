#!/usr/bin/env python3
"""
Test VAE on Cached Test Samples.

Loads cached test samples and evaluates the trained VAE model.
Computes reconstruction loss, KL divergence, and analyzes latent representations.
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys
from datetime import datetime
from typing import Optional, Tuple, Dict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vae_infonce import VAE


def load_test_samples(cache_dir: str) -> Optional[np.ndarray]:
    """Load test samples from cache.
    
    Args:
        cache_dir: Directory containing cached test samples
        
    Returns:
        Test features array or None
    """
    cache_path = Path(cache_dir)
    
    # Find most recent test cache
    test_files = sorted(cache_path.glob("test_samples_*.npz"))
    
    if not test_files:
        print(f"No test cache found in {cache_dir}")
        return None
    
    # Load most recent
    latest = test_files[-1]
    print(f"Loading test cache: {latest}")
    
    data = np.load(latest)
    return data['features']


def load_model(checkpoint_path: str, device: torch.device) -> VAE:
    """Load VAE model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded VAE model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    config = checkpoint.get('config', {})
    input_dim = config.get('input_dim', 80)
    latent_dim = config.get('latent_dim', 16)
    
    # Create model
    model = VAE(input_dim=input_dim, latent_dim=latent_dim, device=device)
    
    # Load state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Input dim: {input_dim}, Latent dim: {latent_dim}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    return model, config


def compute_metrics(
    model: VAE,
    test_features: np.ndarray,
    device: torch.device
) -> Dict[str, float]:
    """Compute evaluation metrics on test set.
    
    Args:
        model: VAE model
        test_features: Test feature array
        device: Device
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Convert to tensor
    test_tensor = torch.FloatTensor(test_features).to(device)
    
    # Encode to latent space
    with torch.no_grad():
        mu, logvar = model.encode(test_tensor)
        
        # Sample from latent
        z = model.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = model.decode(z)
        
        # Compute losses
        recon_loss = torch.nn.functional.binary_cross_entropy(
            reconstruction, test_tensor, reduction='mean'
        ).item()
        
        kl_loss = model.kl_divergence(mu, logvar).item()
        
        # Total loss
        total_loss = recon_loss + kl_loss
        
        # Latent statistics
        latent_mean = mu.mean(dim=0).cpu().numpy()
        latent_std = mu.std(dim=0).cpu().numpy()
        latent_std_overall = z.std(dim=0).mean().item()
        
        # Reconstruction quality
        mse = torch.nn.functional.mse_loss(reconstruction, test_tensor).item()
        
        # Feature-wise reconstruction error
        recon_errors = np.abs(reconstruction.cpu().numpy() - test_features)
        mean_recon_error = recon_errors.mean()
        max_recon_error = recon_errors.max()
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'mse': mse,
        'latent_std': latent_std_overall,
        'latent_mean_mean': np.abs(latent_mean).mean(),
        'latent_mean_std': latent_mean.std(),
        'mean_recon_error': mean_recon_error,
        'max_recon_error': max_recon_error
    }


def analyze_latent_space(
    model: VAE,
    test_features: np.ndarray,
    device: torch.device,
    num_samples: int = 100
) -> Dict:
    """Analyze latent space representations.
    
    Args:
        model: VAE model
        test_features: Test feature array
        device: Device
        num_samples: Number of samples to analyze
        
    Returns:
        Dictionary of latent analysis
    """
    model.eval()
    
    # Take subset
    test_tensor = torch.FloatTensor(test_features[:num_samples]).to(device)
    
    with torch.no_grad():
        mu, logvar = model.encode(test_tensor)
        z = model.reparameterize(mu, logvar)
        
        # Compute pairwise distances in latent space
        mu_np = mu.cpu().numpy()
        
        # Sample statistics
        latent_stats = {
            'mean': mu_np.mean(axis=0).tolist(),
            'std': mu_np.std(axis=0).tolist(),
            'min': mu_np.min(axis=0).tolist(),
            'max': mu_np.max(axis=0).tolist()
        }
        
        # Compute variance per dimension
        dim_vars = mu_np.var(axis=0)
        meaningful_dims = np.sum(dim_vars > 0.01)
        
        # Check for latent collapse (all dims similar variance)
        var_ratio = dim_vars.max() / (dim_vars.min() + 1e-8)
        
        # Cosine similarity between samples (manual computation)
        # Normalize rows
        norms = np.linalg.norm(mu_np, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        mu_normalized = mu_np / norms
        
        # Compute cosine similarity
        similarities = np.dot(mu_normalized, mu_normalized.T)
        
        # Average off-diagonal similarity (should be low for diverse representations)
        mask = np.ones_like(similarities, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_similarity = similarities[mask].mean()
    
    return {
        'meaningful_dimensions': int(meaningful_dims),
        'variance_ratio': float(var_ratio),
        'average_similarity': float(avg_similarity),
        'latent_sample_stats': latent_stats
    }


def save_results(
    metrics: Dict,
    latent_analysis: Dict,
    config: Dict,
    output_path: str
):
    """Save results to JSON.
    
    Args:
        metrics: Evaluation metrics
        latent_analysis: Latent space analysis
        config: Model config
        output_path: Output file path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        else:
            return obj
    
    metrics = convert_to_native(metrics)
    latent_analysis = convert_to_native(latent_analysis)
    config = convert_to_native(config)
    
    results = {
        'timestamp': timestamp,
        'metrics': metrics,
        'latent_analysis': latent_analysis,
        'config': config
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")


def test_vae_cached(
    model_path: str = "models/checkpoints/vae_streaming.pt",
    cache_dir: str = "data/test_cache",
    output_path: Optional[str] = None
) -> Dict:
    """Test VAE on cached test samples.
    
    Args:
        model_path: Path to model checkpoint
        cache_dir: Directory with test cache
        output_path: Output path for results
        
    Returns:
        Dictionary of test results
    """
    # Setup paths
    PROJECT_ROOT = Path(__file__).parent.parent
    model_path = PROJECT_ROOT / model_path
    cache_dir = PROJECT_ROOT / cache_dir
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / "test_results" / f"test_results_{timestamp}.json"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test samples
    test_features = load_test_samples(str(cache_dir))
    
    if test_features is None:
        raise ValueError("No test samples found!")
    
    print(f"Test samples shape: {test_features.shape}")
    
    # Load model
    model, config = load_model(str(model_path), device)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(model, test_features, device)
    
    print("\nTest Metrics:")
    print(f"  Total Loss: {metrics['total_loss']:.4f}")
    print(f"  Reconstruction Loss: {metrics['recon_loss']:.4f}")
    print(f"  KL Divergence: {metrics['kl_loss']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  Latent Std: {metrics['latent_std']:.4f}")
    print(f"  Mean Recon Error: {metrics['mean_recon_error']:.4f}")
    print(f"  Max Recon Error: {metrics['max_recon_error']:.4f}")
    
    # Analyze latent space
    print("\nAnalyzing latent space...")
    latent_analysis = analyze_latent_space(model, test_features, device)
    
    print("\nLatent Space Analysis:")
    print(f"  Meaningful dimensions: {latent_analysis['meaningful_dimensions']}/16")
    print(f"  Variance ratio: {latent_analysis['variance_ratio']:.2f}")
    print(f"  Average similarity: {latent_analysis['average_similarity']:.4f}")
    
    # Interpret results
    print("\n" + "="*60)
    print("Interpretation")
    print("="*60)
    
    if metrics['kl_loss'] < 0.5:
        print("⚠️  KL divergence is low - possible latent collapse")
    elif metrics['kl_loss'] > 10:
        print("⚠️  KL divergence is high - possible KL dominance")
    else:
        print("✓ KL divergence is healthy")
    
    if metrics['latent_std'] < 0.1:
        print("⚠️  Latent std is low - possible collapse")
    else:
        print("✓ Latent variance is meaningful")
    
    if latent_analysis['meaningful_dimensions'] >= 10:
        print("✓ Latent space uses most dimensions meaningfully")
    else:
        print("⚠️  Latent space may be collapsed to fewer dimensions")
    
    if latent_analysis['average_similarity'] < 0.5:
        print("✓ Representations are diverse (low average similarity)")
    else:
        print("⚠️  Representations may be too similar")
    
    # Save results
    save_results(metrics, latent_analysis, config, str(output_path))
    
    return {
        'metrics': metrics,
        'latent_analysis': latent_analysis
    }


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test VAE on cached test samples')
    parser.add_argument('--model', type=str, default='models/checkpoints/vae_streaming.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--cache_dir', type=str, default='data/test_cache',
                        help='Test cache directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results')
    
    args = parser.parse_args()
    
    results = test_vae_cached(
        model_path=args.model,
        cache_dir=args.cache_dir,
        output_path=args.output
    )
    
    return results


if __name__ == "__main__":
    main()
