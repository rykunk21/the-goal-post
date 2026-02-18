"""
Mock data generators for NCAAB VAE prediction testing.

This module provides synthetic data generators for:
- Team feature vectors (80-dim)
- Latent representations (16-dim)
- Transition probabilities (8-dim output)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json


# Constants
FEATURE_DIM = 80
LATENT_DIM = 16
TRANSITION_DIM = 8
NUM_TEAMS = 350  # Approximate number of D1 teams


class TeamFeatureGenerator:
    """Generate synthetic 80-dimensional team feature vectors."""
    
    # Feature categories and their dimensions
    FEATURE_CATEGORIES = {
        'shooting': 20,        # FG%, 3P%, FT%, attempts, makes, etc.
        'rebounding': 12,      # OReb, DReb, TReb, reb rates
        'turnovers': 8,       # TO, TO rate, steals, etc.
        'defense': 12,        # Blocks, steals, defensive ratings
        'playmaking': 10,     # Assists, assist ratios
        'fouls': 6,           # PF, foul rates, drawn fouls
        'pace': 6,            # Possessions, pace factors
        'efficiency': 6        # Offensive/defensive ratings
    }
    
    # Team name mappings for reproducibility
    TEAM_SEED_MAPPING = {
        'Alabama': 0, 'Auburn': 1, 'Kentucky': 2, 'Louisville': 3,
        'Duke': 4, 'North Carolina': 5, 'Kansas': 6, 'UConn': 7,
        'Michigan St.': 8, 'Arizona': 9, 'Houston': 10, 'Marquette': 11
    }
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._feature_bounds = self._init_bounds()
    
    def _init_bounds(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Initialize realistic bounds for each feature category."""
        bounds = {}
        
        # Shooting: percentages [0.25, 0.65], counts [0, 100]
        bounds['shooting'] = (np.array([0.25] * 12 + [0] * 8),
                             np.array([0.65] * 12 + [100] * 8))
        
        # Rebounding: [0, 40]
        bounds['rebounding'] = (np.zeros(12), np.full(12, 40.0))
        
        # Turnovers: [0, 25]
        bounds['turnovers'] = (np.zeros(8), np.full(8, 25.0))
        
        # Defense: blocks [0, 10], steals [0, 15]
        bounds['defense'] = (np.zeros(12), np.array([10, 15, 10, 15, 50, 50, 50, 50, 100, 100, 110, 110]))
        
        # Playmaking: [0, 30]
        bounds['playmaking'] = (np.zeros(10), np.full(10, 30.0))
        
        # Fouls: [0, 30]
        bounds['fouls'] = (np.zeros(6), np.full(6, 30.0))
        
        # Pace: [50, 100]
        bounds['pace'] = (np.full(6, 50.0), np.full(6, 100.0))
        
        # Efficiency: ratings [80, 130]
        bounds['efficiency'] = (np.full(6, 80.0), np.full(6, 130.0))
        
        return bounds
    
    def generate(self, team_id: Optional[str] = None, quality: str = 'average') -> np.ndarray:
        """
        Generate a single team feature vector.
        
        Args:
            team_id: Optional team identifier for seeded generation
            quality: 'strong', 'average', 'weak' to bias the features
            
        Returns:
            80-dimensional numpy array of team features
        """
        if team_id is not None and team_id in self.TEAM_SEED_MAPPING:
            # Use team-specific seed for reproducibility
            team_seed = self.TEAM_SEED_MAPPING[team_id]
            rng = np.random.RandomState(seed + team_seed)
        else:
            rng = self.rng
        
        # Quality modifiers
        quality_mods = {
            'strong': (1.1, 1.25),
            'average': (0.9, 1.1),
            'weak': (0.7, 0.95)
        }
        low_mod, high_mod = quality_mods.get(quality, (0.9, 1.1))
        
        features = []
        for category, dim in self.FEATURE_CATEGORIES.items():
            low, high = self._feature_bounds[category]
            # Generate within bounds with quality modifier
            scale = high - low
            base = rng.rand(dim) * scale + low
            if quality != 'average':
                base = base * rng.uniform(low_mod, high_mod)
            features.extend(base.tolist())
        
        return np.array(features, dtype=np.float32)
    
    def generate_batch(self, n: int, quality_distribution: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Generate multiple team feature vectors.
        
        Args:
            n: Number of teams to generate
            quality_distribution: Dict with keys 'strong', 'average', 'weak' and probabilities
            
        Returns:
            (n, 80) numpy array
        """
        if quality_distribution is None:
            quality_distribution = {'strong': 0.2, 'average': 0.6, 'weak': 0.2}
        
        qualities = list(quality_distribution.keys())
        probs = list(quality_distribution.values())
        
        features = []
        for i in range(n):
            quality = self.rng.choice(qualities, p=probs)
            features.append(self.generate())
        
        return np.array(features, dtype=np.float32)


class LatentRepresentationGenerator:
    """Generate mock latent representations (16-dim vectors for VAE)."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def generate(self, team_id: Optional[str] = None) -> np.ndarray:
        """
        Generate a 16-dimensional latent representation.
        
        The latent space represents:
        - Dimensions 0-3: Offensive capability (shooting, scoring)
        - Dimensions 4-7: Defensive capability
        - Dimensions 8-11: Pace/style factors
        - Dimensions 12-15: Consistency/performance factors
        
        Args:
            team_id: Optional team ID for deterministic generation
            
        Returns:
            16-dimensional numpy array
        """
        if team_id is not None:
            team_seed = hash(team_id) % 10000
            rng = np.random.RandomState(team_seed)
        else:
            rng = self.rng
        
        # Generate latent with realistic structure
        latent = np.zeros(16, dtype=np.float32)
        
        # Offensive (0-3): mean around 0, std 1
        latent[0:4] = rng.randn(4)
        
        # Defensive (4-7): mean around 0, std 1
        latent[4:8] = rng.randn(4)
        
        # Pace (8-11): skewed positive (teams tend to play faster)
        latent[8:12] = rng.randn(4) + 0.3
        
        # Consistency (12-15): mostly positive (teams vary)
        latent[12:16] = np.abs(rng.randn(4)) * 0.8
        
        return latent
    
    def generate_batch(self, n: int, team_ids: Optional[List[str]] = None) -> np.ndarray:
        """Generate multiple latent representations."""
        if team_ids is not None:
            return np.array([self.generate(tid) for tid in team_ids], dtype=np.float32)
        return np.array([self.generate() for _ in range(n)], dtype=np.float32)
    
    def to_transition(self, latent: np.ndarray) -> np.ndarray:
        """
        Convert latent representation to transition probabilities (8-dim).
        
        This simulates what the decoder portion of the VAE would produce.
        
        Args:
            latent: 16-dimensional latent vector
            
        Returns:
            8-dimensional transition probability vector
        """
        # Transform through learned projection (simulated)
        # Each output dimension represents a transition probability
        W = np.random.randn(16, 8).astype(np.float32) * 0.3
        b = np.array([0.1, -0.1, 0.2, 0.0, -0.2, 0.15, 0.05, -0.05], dtype=np.float32)
        
        transition = latent @ W + b
        
        # Apply softmax-like transformation for probabilities
        transition = np.exp(transition - np.max(transition))
        transition = transition / transition.sum()
        
        return transition


class TransitionProbabilityGenerator:
    """Generate mock transition probabilities (8-dim output)."""
    
    # Transition types these probabilities represent
    TRANSITION_TYPES = [
        'win_to_win', 'win_to_loss', 'loss_to_win', 'loss_to_loss',
        'high_scoring', 'low_scoring', 'high_pace', 'low_pace'
    ]
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def generate(self, team_quality: float = 0.5) -> np.ndarray:
        """
        Generate 8-dimensional transition probabilities.
        
        Args:
            team_quality: Float in [0, 1] representing team strength
                         0 = weak team, 1 = strong team
                         
        Returns:
            8-dimensional numpy array (sums to 1)
        """
        # Base probabilities
        probs = np.array([
            0.15,  # win_to_win
            0.10,  # win_to_loss
            0.10,  # loss_to_win
            0.15,  # loss_to_loss
            0.12,  # high_scoring
            0.13,  # low_scoring
            0.12,  # high_pace
            0.13   # low_pace
        ], dtype=np.float32)
        
        # Adjust based on team quality
        # Strong teams: higher win-to-win, higher high_scoring
        # Weak teams: higher loss-to-loss, lower high_scoring
        quality_factor = team_quality - 0.5
        
        probs[0] += quality_factor * 0.1   # win_to_win
        probs[1] -= quality_factor * 0.05   # win_to_loss
        probs[2] += quality_factor * 0.05   # loss_to_win
        probs[3] -= quality_factor * 0.1    # loss_to_loss
        probs[4] += quality_factor * 0.08   # high_scoring
        probs[5] -= quality_factor * 0.08   # low_scoring
        
        # Ensure valid probabilities
        probs = np.clip(probs, 0.02, 0.35)
        probs = probs / probs.sum()
        
        return probs
    
    def generate_batch(self, n: int, quality_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """Generate multiple transition probability vectors."""
        probs_list = []
        for i in range(n):
            quality = self.rng.uniform(*quality_range)
            probs_list.append(self.generate(quality))
        return np.array(probs_list, dtype=np.float32)


class MockVAEData:
    """Combined mock data generator for full VAE pipeline testing."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.feature_gen = TeamFeatureGenerator(seed)
        self.latent_gen = LatentRepresentationGenerator(seed)
        self.transition_gen = TransitionProbabilityGenerator(seed)
    
    def generate_team_data(self, n_teams: int = 100) -> Dict:
        """
        Generate complete mock dataset for VAE training/testing.
        
        Returns:
            Dictionary with keys:
            - features: (n_teams, 80) team feature vectors
            - latent: (n_teams, 16) latent representations
            - transitions: (n_teams, 8) transition probabilities
            - team_ids: list of team identifiers
        """
        features = self.feature_gen.generate_batch(n_teams)
        
        # Generate latent from features (simulated encoder)
        # In real VAE, this would be the encoder network
        latent = self.latent_gen.generate_batch(n_teams)
        
        # Generate transitions from latent (simulated decoder)
        transitions = np.array([
            self.latent_gen.to_transition(l) for l in latent
        ], dtype=np.float32)
        
        # Generate team IDs
        team_ids = [f"team_{i:04d}" for i in range(n_teams)]
        
        return {
            'features': features,
            'latent': latent,
            'transitions': transitions,
            'team_ids': team_ids,
            'feature_dim': FEATURE_DIM,
            'latent_dim': LATENT_DIM,
            'transition_dim': TRANSITION_DIM
        }
    
    def generate_game_pair(self, team_a_id: str, team_b_id: str) -> Tuple[Dict, Dict]:
        """
        Generate game data for two teams.
        
        Returns two dictionaries, one for each team containing:
        - features: 80-dim feature vector
        - latent: 16-dim latent rep
        - transitions: 8-dim transition probs
        """
        team_a = {
            'team_id': team_a_id,
            'features': self.feature_gen.generate(team_a_id),
            'latent': self.latent_gen.generate(team_a_id),
        }
        team_a['transitions'] = self.latent_gen.to_transition(team_a['latent'])
        
        team_b = {
            'team_id': team_b_id,
            'features': self.feature_gen.generate(team_b_id),
            'latent': self.latent_gen.generate(team_b_id),
        }
        team_b['transitions'] = self.latent_gen.to_transition(team_b['latent'])
        
        return team_a, team_b
    
    def save(self, filepath: str, data: Dict):
        """Save mock data to JSON file."""
        # Convert numpy arrays to lists for JSON
        json_data = {
            'features': data['features'].tolist(),
            'latent': data['latent'].tolist(),
            'transitions': data['transitions'].tolist(),
            'team_ids': data['team_ids'],
            'metadata': {
                'feature_dim': data['feature_dim'],
                'latent_dim': data['latent_dim'],
                'transition_dim': data['transition_dim']
            }
        }
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> Dict:
        """Load mock data from JSON file."""
        with open(filepath, 'r') as f:
            json_data = json.load(f)
        
        return {
            'features': np.array(json_data['features'], dtype=np.float32),
            'latent': np.array(json_data['latent'], dtype=np.float32),
            'transitions': np.array(json_data['transitions'], dtype=np.float32),
            'team_ids': json_data['team_ids'],
            'feature_dim': json_data['metadata']['feature_dim'],
            'latent_dim': json_data['metadata']['latent_dim'],
            'transition_dim': json_data['metadata']['transition_dim']
        }


def generate_test_data(n_teams: int = 50) -> Dict:
    """Convenience function to generate test data."""
    mock_vae = MockVAEData(seed=12345)
    return mock_vae.generate_team_data(n_teams)


if __name__ == '__main__':
    # Demo usage
    print("Generating mock VAE data...")
    
    # Create generator
    mock = MockVAEData(seed=42)
    
    # Generate team data
    data = mock.generate_team_data(n_teams=20)
    
    print(f"Features shape: {data['features'].shape}")
    print(f"Latent shape: {data['latent'].shape}")
    print(f"Transitions shape: {data['transitions'].shape}")
    print(f"\nSample team IDs: {data['team_ids'][:5]}")
    print(f"\nSample feature vector (first 10 dims):")
    print(data['features'][0][:10])
    print(f"\nSample latent vector:")
    print(data['latent'][0])
    print(f"\nSample transition probs (sum={data['transitions'][0].sum():.4f}):")
    print(data['transitions'][0])
    
    # Save to file
    mock.save('/tmp/mock_vae_data.json', data)
    print("\nSaved to /tmp/mock_vae_data.json")
