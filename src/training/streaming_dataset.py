"""
Streaming Dataset for NCAAB Prediction System.

PyTorch IterableDataset that streams game data from StatBroadcast,
splits into train/test (80/20), and caches test set.
"""

import os
import random
import time
from typing import List, Optional, Iterator, Tuple
import numpy as np
import torch
from torch.utils.data import IterableDataset
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingTeamDataset(IterableDataset):
    """Streaming dataset for NCAAB team features.
    
    Features:
        - Streams game data from StatBroadcast
        - 80/20 train/test split with probability
        - Caches test set to disk with timestamp
        - Yields training samples immediately
        - Rate limited fetching
        
    Args:
        game_ids: List of game IDs to stream
        test_prob: Probability of test split (default 0.2)
        cache_dir: Directory for test cache
        shuffle: Whether to shuffle game IDs
    """
    
    def __init__(
        self,
        game_ids: List[int],
        test_prob: float = 0.2,
        cache_dir: str = "data/test_cache",
        shuffle: bool = True,
        seed: int = 42
    ):
        """Initialize streaming dataset.
        
        Args:
            game_ids: List of game IDs to process
            test_prob: Probability of test split
            cache_dir: Directory to cache test samples
            shuffle: Whether to shuffle game order
            seed: Random seed for reproducibility
        """
        self.game_ids = list(game_ids)
        self.test_prob = test_prob
        self.cache_dir = Path(cache_dir)
        self.shuffle = shuffle
        self.seed = seed
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track test samples for caching
        self._test_samples = []
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._test_cache_path = self.cache_dir / f"test_samples_{self._timestamp}.npz"
        
        # Statistics
        self._games_processed = 0
        self._games_success = 0
        self._test_samples_count = 0
        self._train_samples_count = 0
        
        # Set random seed
        random.seed(seed)
    
    def _init_loader(self):
        """Initialize the streaming loader."""
        from src.data.streaming_loader import StreamingXMLoader
        return StreamingXMLoader()
    
    def _should_be_test(self) -> bool:
        """Determine if current sample should be test set.
        
        Returns:
            True if should be test set
        """
        return random.random() < self.test_prob
    
    def _cache_test_sample(self, features: np.ndarray):
        """Cache a test sample to disk.
        
        Args:
            features: Feature vector (80-dim)
        """
        self._test_samples.append(features.copy())
        self._test_samples_count += 1
        
        # Save periodically (every 10 samples)
        if self._test_samples_count % 10 == 0:
            self._save_test_cache()
    
    def _save_test_cache(self):
        """Save test cache to disk."""
        if not self._test_samples:
            return
        
        # Convert to numpy array
        test_data = np.array(self._test_samples)
        
        # Save with timestamp
        np.savez(
            self._test_cache_path,
            features=test_data,
            timestamp=self._timestamp,
            num_samples=len(self._test_samples)
        )
        
        logger.info(f"Cached {len(self._test_samples)} test samples to {self._test_cache_path}")
    
    def _load_existing_test_cache(self) -> Optional[np.ndarray]:
        """Load existing test cache if available.
        
        Returns:
            Test features array or None
        """
        # Find most recent test cache
        test_files = sorted(self.cache_dir.glob("test_samples_*.npz"))
        
        if not test_files:
            return None
        
        # Load most recent
        latest = test_files[-1]
        logger.info(f"Loading existing test cache: {latest}")
        
        data = np.load(latest)
        return data['features']
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, str]]:
        """Iterate over game IDs, fetching and splitting data.
        
        Yields:
            Tuple of (features tensor, set_type string)
        """
        # Shuffle game IDs if requested
        game_ids = self.game_ids.copy()
        if self.shuffle:
            random.shuffle(game_ids)
        
        # Initialize loader
        loader = self._init_loader()
        
        for game_id in game_ids:
            self._games_processed += 1
            
            try:
                # Fetch features
                home_features, away_features = loader.fetch_game_features(game_id)
                
                if home_features is None or away_features is None:
                    logger.debug(f"Skipping game {game_id} (fetch failed)")
                    continue
                
                self._games_success += 1
                
                # Process home team
                if self._should_be_test():
                    self._cache_test_sample(home_features)
                    set_type = "test"
                else:
                    yield torch.FloatTensor(home_features), "train"
                    self._train_samples_count += 1
                    set_type = "train"
                
                # Process away team
                if self._should_be_test():
                    self._cache_test_sample(away_features)
                else:
                    yield torch.FloatTensor(away_features), "train"
                    self._train_samples_count += 1
                
                # Log progress
                if self._games_processed % 50 == 0:
                    logger.info(
                        f"Processed {self._games_processed} games, "
                        f"success: {self._games_success}, "
                        f"train: {self._train_samples_count}, "
                        f"test cached: {self._test_samples_count}"
                    )
                
            except Exception as e:
                logger.warning(f"Error processing game {game_id}: {e}")
                continue
        
        # Final save of test cache
        self._save_test_cache()
        
        logger.info(
            f"Streaming complete: {self._games_processed} processed, "
            f"{self._games_success} successful, "
            f"{self._train_samples_count} train, "
            f"{self._test_samples_count} test"
        )
    
    def get_stats(self) -> dict:
        """Get dataset statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'games_processed': self._games_processed,
            'games_success': self._games_success,
            'train_samples': self._train_samples_count,
            'test_samples': self._test_samples_count,
            'test_cache_path': str(self._test_cache_path),
            'timestamp': self._timestamp
        }


def create_streaming_dataloaders(
    game_ids: List[int],
    batch_size: int = 32,
    test_prob: float = 0.2,
    cache_dir: str = "data/test_cache",
    num_workers: int = 0
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and test dataloaders from streaming dataset.
    
    Note: Since IterableDataset doesn't support random split,
    we use probability-based splitting and yield training samples
    while caching test samples.
    
    Args:
        game_ids: List of game IDs to stream
        batch_size: Batch size
        test_prob: Probability for test split
        cache_dir: Directory for test cache
        num_workers: Number of workers (0 for streaming)
        
    Returns:
        Tuple of (train_loader, test_features)
    """
    from torch.utils.data import DataLoader
    
    # Create streaming dataset
    dataset = StreamingTeamDataset(
        game_ids=game_ids,
        test_prob=test_prob,
        cache_dir=cache_dir,
        shuffle=True
    )
    
    # For training, filter to only train samples
    # Since IterableDataset yields as it goes, we need a different approach
    # We'll collect train samples into a list for DataLoader
    
    train_samples = []
    
    for features, set_type in dataset:
        if set_type == "train":
            train_samples.append(features)
    
    # Get test samples from cache
    test_cache_path = Path(cache_dir)
    test_files = sorted(test_cache_path.glob("test_samples_*.npz"))
    
    test_features = None
    if test_files:
        latest = test_files[-1]
        test_data = np.load(latest)
        test_features = torch.FloatTensor(test_data['features'])
    
    # Create DataLoaders
    if train_samples:
        train_dataset = torch.utils.data.TensorDataset(
            torch.stack(train_samples)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
    else:
        train_loader = None
    
    return train_loader, test_features, dataset.get_stats()


# Convenience function for simpler use case
def get_streaming_dataloader(
    game_ids: List[int],
    batch_size: int = 32,
    test_prob: float = 0.2,
    cache_dir: str = "data/test_cache"
) -> Iterator:
    """Get a streaming dataloader (train samples only).
    
    Args:
        game_ids: List of game IDs
        batch_size: Batch size
        test_prob: Test probability
        cache_dir: Test cache directory
        
    Yields:
        Batches of training data
    """
    dataset = StreamingTeamDataset(
        game_ids=game_ids,
        test_prob=test_prob,
        cache_dir=cache_dir,
        shuffle=True
    )
    
    # Collect training samples
    train_samples = []
    
    for features, set_type in dataset:
        if set_type == "train":
            train_samples.append(features)
    
    # Yield batches
    for i in range(0, len(train_samples), batch_size):
        batch = train_samples[i:i+batch_size]
        yield torch.stack(batch)


# Test function
def test_streaming_dataset():
    """Test the streaming dataset with a few game IDs."""
    from src.data.game_discovery import generate_game_ids_for_streaming
    
    print("=" * 60)
    print("Streaming Dataset Test")
    print("=" * 60)
    
    # Generate small set of game IDs
    game_ids = generate_game_ids_for_streaming(50, strategy="sequential")
    
    print(f"\nGenerated {len(game_ids)} game IDs")
    print(f"First 10: {game_ids[:10]}")
    
    # Create dataset
    dataset = StreamingTeamDataset(
        game_ids=game_ids,
        test_prob=0.2,
        cache_dir="data/test_cache",
        shuffle=True
    )
    
    print("\nIterating through dataset...")
    
    train_count = 0
    test_count = 0
    
    for features, set_type in dataset:
        if set_type == "train":
            train_count += 1
        else:
            test_count += 1
    
    print(f"\nResults:")
    print(f"  Train samples: {train_count}")
    print(f"  Test samples: {test_count}")
    print(f"  Stats: {dataset.get_stats()}")


if __name__ == "__main__":
    test_streaming_dataset()
