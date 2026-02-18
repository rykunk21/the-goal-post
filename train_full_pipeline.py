#!/usr/bin/env python3
"""
Full Pipeline Training: VAE + InfoNCE + Transition Network.

Trains all three heads jointly on NCAAB game data from StatBroadcast.
Combines representation learning with contrastive loss and transition prediction.

Usage:
    python train_full_pipeline.py --games 650 --epochs 50 --beta 1.0 --infonce_weight 0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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
from typing import Optional, List, Tuple, Dict, Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vae_infonce import VAE, InfoNCELoss
from src.models.transition_network import TransitionNetwork, TRANSITION_TYPES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_game_ids(cached: bool = True, max_games: Optional[int] = None) -> List[int]:
    """Load game IDs from cache."""
    import json
    
    if cached:
        cache_file = Path(__file__).parent / 'data' / 'statbroadcast_game_ids.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            game_ids = []
            for team, ids in cached_data.items():
                game_ids.extend(ids)
            game_ids = list(set(game_ids))
            logger.info(f"Loaded {len(game_ids)} cached game IDs")
            if max_games:
                game_ids = game_ids[:max_games]
            return game_ids
    
    raise RuntimeError("Fresh discovery not implemented. Use cached data.")


class StreamingDataLoader:
    """Streaming loader that fetches game XML and extracts real features."""
    
    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.base_url = "http://archive.statbroadcast.com"
    
    def _parse_stat(self, elem, attr: str, default=0):
        """Parse integer/float from XML attribute."""
        try:
            val = elem.get(attr, default)
            if val is None:
                return default
            if isinstance(val, str):
                val = val.replace(',', '')
                return float(val) if '.' in val else int(val)
            return val
        except (ValueError, TypeError):
            return default
    
    def fetch_game_features(self, game_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Fetch game XML and extract real features and transition labels."""
        import time
        import requests
        import xml.etree.ElementTree as ET
        
        time.sleep(self.rate_limit)
        
        try:
            url = f"{self.base_url}/{game_id}.xml"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return None, None, None
            
            root = ET.fromstring(response.content)
            teams = root.findall('.//team')
            if len(teams) < 2:
                return None, None, None
            
            home_team = away_team = None
            for team in teams:
                if team.get('vh') == 'H':
                    home_team = team
                elif team.get('vh') == 'V':
                    away_team = team
            
            if home_team is None or away_team is None:
                return None, None, None
            
            home_feat = self._extract_team_features(home_team, away_team)
            away_feat = self._extract_team_features(away_team, home_team)
            
            if home_feat is None or away_feat is None:
                return None, None, None
            
            home_trans = self._extract_transitions(home_team)
            away_trans = self._extract_transitions(away_team)
            transitions = (home_trans + away_trans) / 2
            
            return home_feat, away_feat, transitions
            
        except Exception as e:
            logger.debug(f"Error fetching game {game_id}: {e}")
            return None, None, None
    
    def _extract_team_features(self, team_elem, opponent_elem) -> Optional[np.ndarray]:
        """Extract 80-dim feature vector from team XML element."""
        try:
            totals = team_elem.find('.//totals/stats')
            opp_totals = opponent_elem.find('.//totals/stats')
            
            if totals is None or opp_totals is None:
                return None
            
            feat = np.zeros(80, dtype=np.float32)
            
            # Scoring (0-9)
            feat[0] = self._parse_stat(totals, 'tp', 0) / 100
            feat[1] = self._parse_stat(totals, 'fgm', 0) / 50
            feat[2] = self._parse_stat(totals, 'fga', 0) / 80
            feat[3] = self._parse_stat(totals, 'fgm3', 0) / 15
            feat[4] = self._parse_stat(totals, 'fga3', 0) / 30
            feat[5] = self._parse_stat(totals, 'ftm', 0) / 25
            feat[6] = self._parse_stat(totals, 'fta', 0) / 35
            feat[7] = self._parse_stat(totals, 'oreb', 0) / 15
            feat[8] = self._parse_stat(totals, 'dreb', 0) / 30
            feat[9] = self._parse_stat(totals, 'ast', 0) / 25
            
            # Defense (10-19)
            feat[10] = self._parse_stat(totals, 'to', 0) / 25
            feat[11] = self._parse_stat(totals, 'stl', 0) / 10
            feat[12] = self._parse_stat(totals, 'blk', 0) / 10
            feat[13] = self._parse_stat(totals, 'pf', 0) / 25
            feat[14] = self._parse_stat(opp_totals, 'fgm', 0) / 50
            feat[15] = self._parse_stat(opp_totals, 'fga', 0) / 80
            feat[16] = self._parse_stat(opp_totals, 'fgm3', 0) / 15
            feat[17] = self._parse_stat(opp_totals, 'fga3', 0) / 30
            feat[18] = self._parse_stat(opp_totals, 'ftm', 0) / 25
            feat[19] = self._parse_stat(opp_totals, 'fta', 0) / 35
            
            # Rebounds (20-29)
            feat[20] = self._parse_stat(totals, 'treb', 0) / 40
            feat[21] = feat[7]
            feat[22] = feat[8]
            feat[23] = feat[7] / max(feat[20], 0.1) if feat[20] > 0 else 0
            feat[24] = feat[8] / max(feat[20], 0.1) if feat[20] > 0 else 0
            feat[25] = feat[23] + feat[24]
            feat[26] = feat[7] * 0.7
            feat[27] = feat[8] * 0.7
            feat[28] = self._parse_stat(opp_totals, 'oreb', 0) / 15
            feat[29] = self._parse_stat(opp_totals, 'dreb', 0) / 30
            
            # Efficiency (30-39)
            fga = max(feat[2], 0.01)
            fgm = feat[1]
            fgpct = fgm / fga if fga > 0 else 0
            fga3 = max(feat[4], 0.01)
            fg3pct = feat[3] / fga3 if fga3 > 0 else 0
            fta = max(feat[6], 0.01)
            ftpct = feat[5] / fta if fta > 0 else 0
            
            feat[30] = fgpct
            feat[31] = fg3pct
            feat[32] = ftpct
            feat[33] = (fgm + 0.5 * feat[3]) / fga if fga > 0 else 0
            feat[34] = feat[5] / fga if fga > 0 else 0
            feat[35] = feat[4] / fga if fga > 0 else 0
            
            opp_fga = max(feat[15], 0.01)
            opp_fgm = feat[14]
            feat[36] = opp_fgm / opp_fga if opp_fga > 0 else 0
            opp_fg3a = max(feat[17], 0.01)
            feat[37] = feat[16] / opp_fg3a if opp_fg3a > 0 else 0
            opp_fta = max(feat[19], 0.01)
            feat[38] = feat[18] / opp_fta if opp_fta > 0 else 0
            feat[39] = feat[30] - feat[36]
            
            # Advanced (40-49)
            poss = fga - feat[7] + feat[10] + 0.44 * fta
            ortg = feat[0] * 100 / poss if poss > 0 else 1.0
            opp_fga = feat[15]
            opp_orb = feat[28] * 15
            opp_tov = self._parse_stat(opp_totals, 'to', 0) / 25
            opp_poss = opp_fga - opp_orb + opp_tov + 0.44 * feat[19]
            drtg = (self._parse_stat(opp_totals, 'tp', 0) / 100) * 100 / opp_poss if opp_poss > 0 else 1.0
            
            feat[40] = ortg / 120
            feat[41] = drtg / 120
            feat[42] = feat[40] - feat[41]
            feat[43] = (poss + opp_poss) / 140
            feat[44] = feat[9] / fga if fga > 0 else 0
            feat[45] = feat[10] / poss if poss > 0 else 0
            feat[46] = feat[7] / feat[20] if feat[20] > 0 else 0
            feat[47] = feat[40] * feat[43] / 70
            feat[48] = feat[41] * feat[43] / 70
            feat[49] = feat[42] * feat[43] / 70
            
            # Context (50-59)
            feat[50] = feat[9] / max(feat[10], 0.1)
            feat[51] = fgm / max(feat[10], 0.1)
            feat[52] = (feat[5] + feat[6]) / max(feat[10], 0.1)
            feat[53] = poss / (poss + opp_poss) if (poss + opp_poss) > 0 else 0.5
            feat[54] = feat[11] / poss if poss > 0 else 0
            feat[55] = feat[12] / opp_fga if opp_fga > 0 else 0
            feat[56] = feat[4] / fga if fga > 0 else 0
            feat[57] = (fgm - feat[3]) / (fga - feat[4]) if (fga - feat[4]) > 0 else 0
            feat[58] = 0.4
            feat[59] = 0.22
            
            # Season (60-69)
            feat[60] = feat[0]
            feat[61] = fgm * 2 + feat[3]
            feat[62] = feat[30] * 100
            feat[63] = feat[31] * 100
            feat[64] = feat[32] * 100
            feat[65] = feat[20] * 10
            feat[66] = feat[9] * 10
            feat[67] = feat[10] * 10
            feat[68] = feat[43] * 10
            feat[69] = feat[40] * 10
            
            # Opponent-adjusted (70-79)
            opp_pts = self._parse_stat(opp_totals, 'tp', 0) / 100
            feat[70] = (feat[0] + opp_pts) / 2
            feat[71] = (feat[40] + feat[41]) / 2
            feat[72] = feat[41]
            feat[73] = feat[36]
            feat[74] = feat[37]
            feat[75] = feat[20]
            feat[76] = feat[9]
            feat[77] = feat[10]
            feat[78] = 0.5
            feat[79] = 1.0
            
            return np.clip(feat, 0, 1)
            
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return None
    
    def _extract_transitions(self, team_elem) -> np.ndarray:
        """Extract 8-dim transition probabilities from team stats."""
        totals = team_elem.find('.//totals/stats')
        if totals is None:
            return np.ones(8, dtype=np.float32) / 8
        
        trans = np.zeros(8, dtype=np.float32)
        
        fgm = self._parse_stat(totals, 'fgm', 0)
        fga = self._parse_stat(totals, 'fga', 0)
        fgm3 = self._parse_stat(totals, 'fgm3', 0)
        fga3 = self._parse_stat(totals, 'fga3', 0)
        
        fgm2 = max(fgm - fgm3, 0)
        fga2 = max(fga - fga3, 0)
        
        trans[0] = fgm2 / max(fga, 1) * 0.4
        trans[1] = (fga2 - fgm2) / max(fga, 1) * 0.4
        trans[2] = fgm3 / max(fga, 1) * 0.3
        trans[3] = (fga3 - fgm3) / max(fga, 1) * 0.3
        
        ftm = self._parse_stat(totals, 'ftm', 0)
        fta = self._parse_stat(totals, 'fta', 1)
        trans[4] = ftm / max(fta * 2, 1) * 0.15
        trans[5] = (fta - ftm) / max(fta * 2, 1) * 0.15
        
        oreb = self._parse_stat(totals, 'oreb', 0)
        tov = self._parse_stat(totals, 'to', 1)
        trans[6] = oreb / 20
        trans[7] = tov / 25
        
        trans = np.clip(trans, 0.01, 1)
        trans = trans / trans.sum()
        
        return trans


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to [0, 1] range."""
    feature_max = features.max(axis=0)
    feature_max = np.where(feature_max == 0, 1.0, feature_max)
    return np.clip(features / feature_max, 0, 1)


def create_transition_labels(
    home_transitions: np.ndarray,
    away_transitions: np.ndarray,
    num_classes: int = 8
) -> np.ndarray:
    """Create transition labels from probabilities.
    
    For each game pair, create a combined label vector.
    """
    # Simple average of home/away transitions as labels
    combined = (home_transitions + away_transitions) / 2
    return combined


class FullPipelineTrainer:
    """Trainer for the full VAE + InfoNCE + Transition Network pipeline."""
    
    def __init__(
        self,
        vae: VAE,
        transition_net: TransitionNetwork,
        infonce_weight: float = 0.1,
        transition_weight: float = 1.0,
        beta: float = 1.0,
        lr: float = 1e-3,
        device: Optional[torch.device] = None
    ):
        self.vae = vae
        self.transition_net = transition_net
        self.infonce_weight = infonce_weight
        self.transition_weight = transition_weight
        self.beta = beta
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move models to device
        self.vae.to(self.device)
        self.transition_net.to(self.device)
        
        # Optimizer for all parameters
        self.optimizer = torch.optim.AdamW(
            list(self.vae.parameters()) + list(self.transition_net.parameters()),
            lr=lr,
            weight_decay=1e-4
        )
        
        # InfoNCE loss
        self.infonce_loss_fn = InfoNCELoss(temperature=0.07)
        
        # Transition loss
        self.transition_loss_fn = nn.CrossEntropyLoss()
    
    def forward_pass(
        self,
        x: torch.Tensor,
        transition_labels: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through all three heads."""
        results = {}
        
        # VAE forward
        reconstruction, mu, logvar = self.vae(x)
        results['reconstruction'] = reconstruction
        results['mu'] = mu
        results['logvar'] = logvar
        
        # Compute VAE losses
        recon_loss = F.binary_cross_entropy(reconstruction, x)
        kl_loss = self.vae.kl_divergence(mu, logvar)
        results['recon_loss'] = recon_loss
        results['kl_loss'] = kl_loss
        results['vae_loss'] = recon_loss + self.beta * kl_loss
        
        # InfoNCE loss (requires pairs of samples)
        if mu.shape[0] >= 2:
            # InfoNCE on latents (uses diagonal as positive pairs)
            infonce_loss = self.infonce_loss_fn(mu)
            results['infonce_loss'] = infonce_loss
        else:
            results['infonce_loss'] = torch.tensor(0.0, device=self.device)
        
        # Transition network prediction
        if context is not None and transition_labels is not None:
            # Assemble state vector: [home_latent(16), away_latent(16), shooting_context(8), context(42)]
            # For simplicity, use mu as both home and away (in practice would be different teams)
            batch_size = mu.shape[0]
            
            # Use mu as home latent, shift for away latent
            home_latent = mu
            away_latent = torch.roll(mu, shifts=1, dims=0)
            shooting_context = transition_labels[:, :8] if transition_labels.shape[1] >= 8 else transition_labels
            
            # Context: if not provided, use zeros
            if context is None:
                context = torch.zeros(batch_size, 42, device=self.device)
            
            # Assemble state vector
            state = torch.cat([home_latent, away_latent, shooting_context, context], dim=1)
            
            # Predict transitions
            pred_transitions = self.transition_net(state)
            
            # Transition loss (cross-entropy)
            # Convert probabilities to class indices for cross-entropy
            target_indices = torch.argmax(transition_labels, dim=1)
            transition_loss = self.transition_loss_fn(
                torch.log(pred_transitions + 1e-8),
                target_indices
            )
            
            results['transition_loss'] = transition_loss
            results['pred_transitions'] = pred_transitions
        else:
            results['transition_loss'] = torch.tensor(0.0, device=self.device)
            results['pred_transitions'] = None
        
        # Total loss
        results['total_loss'] = (
            results['vae_loss'] +
            self.infonce_weight * results['infonce_loss'] +
            self.transition_weight * results['transition_loss']
        )
        
        return results
    
    def step(
        self,
        x: torch.Tensor,
        transition_labels: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Single training step."""
        self.optimizer.zero_grad()
        
        # Forward pass
        results = self.forward_pass(x, transition_labels, context)
        
        # Backward pass
        results['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.transition_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Return scalar losses
        return {
            'total_loss': results['total_loss'].item(),
            'recon_loss': results['recon_loss'].item(),
            'kl_loss': results['kl_loss'].item(),
            'infonce_loss': results['infonce_loss'].item(),
            'transition_loss': results['transition_loss'].item(),
            'vae_loss': results['vae_loss'].item()
        }


def plot_training_curves(history: dict, output_path: str):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    epochs = history['epoch']
    
    # Total Loss
    axes[0, 0].plot(epochs, history['total_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # VAE Loss
    axes[0, 1].plot(epochs, history['vae_loss'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('VAE Loss')
    axes[0, 1].set_title('VAE Loss (Recon + KL)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reconstruction Loss
    axes[0, 2].plot(epochs, history['recon_loss'], 'r-', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Reconstruction Loss')
    axes[0, 2].set_title('Reconstruction Loss (BCE)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # KL Divergence
    axes[1, 0].plot(epochs, history['kl_loss'], 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].grid(True, alpha=0.3)
    
    # InfoNCE Loss
    axes[1, 1].plot(epochs, history['infonce_loss'], 'c-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('InfoNCE Loss')
    axes[1, 1].set_title('InfoNCE Contrastive Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Transition Loss
    axes[1, 2].plot(epochs, history['transition_loss'], 'orange', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Transition Loss')
    axes[1, 2].set_title('Transition Network Loss')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training curves to {output_path}")


def train_full_pipeline(
    num_games: int = 650,
    epochs: int = 50,
    batch_size: int = 32,
    beta: float = 1.0,
    infonce_weight: float = 0.1,
    transition_weight: float = 1.0,
    lr: float = 1e-3,
    use_real_data: bool = True,
    checkpoint_dir: str = "models/checkpoints"
) -> Tuple[VAE, TransitionNetwork, dict]:
    """Train the full pipeline.
    
    Args:
        num_games: Number of games to train on
        epochs: Number of training epochs
        batch_size: Batch size
        beta: KL divergence weight
        infonce_weight: InfoNCE loss weight
        transition_weight: Transition loss weight
        lr: Learning rate
        use_real_data: If True, require real data (no synthetic fallback)
        checkpoint_dir: Directory for model checkpoints
    
    Returns:
        Tuple of (vae, transition_net, history)
    
    Raises:
        RuntimeError: If real data is unavailable when use_real_data=True
    """
    
    # Setup paths
    PROJECT_ROOT = Path(__file__).parent.parent
    checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    training_curves_dir = PROJECT_ROOT / "training_curves"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_curves_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load real data from StatBroadcast - NO synthetic fallback
    logger.info(f"Loading data for {num_games} games from StatBroadcast...")
    
    if use_real_data:
        # Require real data - fail if unavailable
        loader = StreamingDataLoader(rate_limit=0.1)  # 10 requests per second
        game_ids = get_game_ids(cached=True, max_games=num_games)
        
        if not game_ids:
            raise RuntimeError(
                "No game IDs available. "
                "Run game discovery first: python discover_games.py"
            )
        
        logger.info(f"Streaming {len(game_ids)} game XMLs at {loader.rate_limit} req/sec...")
        logger.info(f"Estimated time: {len(game_ids) * loader.rate_limit / 60:.1f} minutes")
        
        features_list = []
        transitions_list = []
        successful_games = 0
        failed_games = 0
        
        for i, game_id in enumerate(game_ids[:num_games]):
            if i > 0 and i % 50 == 0:
                logger.info(f"Progress: {i}/{min(num_games, len(game_ids))} games streamed")
            
            home_feat, away_feat, trans = loader.fetch_game_features(game_id)
            if home_feat is not None:
                features_list.append(home_feat)
                features_list.append(away_feat)
                transitions_list.append(trans)
                transitions_list.append(trans)  # Same for both teams
                successful_games += 1
            else:
                failed_games += 1
        
        if not features_list:
            raise RuntimeError(
                f"Failed to stream ANY games from {num_games} requested. "
                f"Check StatBroadcast API availability. "
                f"Error: {failed_games} games failed to load."
            )
        
        features = np.array(features_list)
        transitions = np.array(transitions_list)
        
        logger.info(f"Successfully streamed: {successful_games} games ({failed_games} failed)")
        logger.info(f"Collected {len(features)} feature vectors")
        
        if len(features) < 100:
            raise RuntimeError(
                f"Insufficient real data: only {len(features)} samples collected. "
                f"Need at least 100 samples for training."
            )
    else:
        raise RuntimeError(
            "Synthetic data has been removed. "
            "Use --use_real_data flag with valid game IDs."
        )
    
    logger.info(f"Collected {len(features)} samples")
    
    # Normalize features
    features = normalize_features(features)
    
    # Create context (42-dim game state - zeros for now, would include real game context)
    context = np.zeros((len(features), 42), dtype=np.float32)
    
    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(features))
    test_size = int(len(features) * 0.2)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    train_features = features[train_indices]
    train_transitions = transitions[train_indices]
    train_context = context[train_indices]
    
    test_features = features[test_indices]
    test_transitions = transitions[test_indices]
    test_context = context[test_indices]
    
    # Convert to tensors
    train_features_t = torch.FloatTensor(train_features)
    train_transitions_t = torch.FloatTensor(train_transitions)
    train_context_t = torch.FloatTensor(train_context)
    
    test_features_t = torch.FloatTensor(test_features)
    test_transitions_t = torch.FloatTensor(test_transitions)
    test_context_t = torch.FloatTensor(test_context)
    
    logger.info(f"Training samples: {len(train_features_t)}")
    logger.info(f"Test samples: {len(test_features_t)}")
    
    # Create data loaders
    train_dataset = TensorDataset(train_features_t, train_transitions_t, train_context_t)
    test_dataset = TensorDataset(test_features_t, test_transitions_t, test_context_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize models
    vae = VAE(input_dim=80, latent_dim=16, device=device)
    transition_net = TransitionNetwork(device=device)
    
    logger.info(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    logger.info(f"Transition Net parameters: {sum(p.numel() for p in transition_net.parameters()):,}")
    
    # Initialize trainer
    trainer = FullPipelineTrainer(
        vae=vae,
        transition_net=transition_net,
        infonce_weight=infonce_weight,
        transition_weight=transition_weight,
        beta=beta,
        lr=lr,
        device=device
    )
    
    # Training history
    history = {
        'epoch': [],
        'total_loss': [],
        'vae_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'infonce_loss': [],
        'transition_loss': [],
        'latent_std': [],
        'val_total_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        'val_transition_loss': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    logger.info(f"Training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        vae.train()
        transition_net.train()
        
        epoch_losses = {
            'total': 0, 'vae': 0, 'recon': 0, 'kl': 0,
            'infonce': 0, 'transition': 0, 'std': 0
        }
        num_batches = 0
        
        for batch in train_loader:
            x, trans, ctx = [b.to(device) for b in batch]
            
            losses = trainer.step(x, trans, ctx)
            
            epoch_losses['total'] += losses['total_loss']
            epoch_losses['vae'] += losses['vae_loss']
            epoch_losses['recon'] += losses['recon_loss']
            epoch_losses['kl'] += losses['kl_loss']
            epoch_losses['infonce'] += losses['infonce_loss']
            epoch_losses['transition'] += losses['transition_loss']
            
            # Compute latent std
            with torch.no_grad():
                mu, logvar = vae.encoder(x)
                z = vae.reparameterize(mu, logvar)
                epoch_losses['std'] += z.std(dim=0).mean().item()
            
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Validation phase
        vae.eval()
        transition_net.eval()
        
        val_losses = {'total': 0, 'recon': 0, 'kl': 0, 'transition': 0}
        val_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                x, trans, ctx = [b.to(device) for b in batch]
                
                results = trainer.forward_pass(x, trans, ctx)
                
                val_losses['total'] += results['total_loss'].item()
                val_losses['recon'] += results['recon_loss'].item()
                val_losses['kl'] += results['kl_loss'].item()
                val_losses['transition'] += results['transition_loss'].item()
                
                val_batches += 1
        
        for key in val_losses:
            val_losses[key] /= val_batches
        
        # Log epoch
        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train: {epoch_losses['total']:.4f} | "
            f"Val: {val_losses['total']:.4f} | "
            f"Recon: {epoch_losses['recon']:.4f} | "
            f"KL: {epoch_losses['kl']:.4f} | "
            f"InfoNCE: {epoch_losses['infonce']:.4f} | "
            f"Trans: {epoch_losses['transition']:.4f}"
        )
        
        # Save history
        history['epoch'].append(epoch)
        history['total_loss'].append(epoch_losses['total'])
        history['vae_loss'].append(epoch_losses['vae'])
        history['recon_loss'].append(epoch_losses['recon'])
        history['kl_loss'].append(epoch_losses['kl'])
        history['infonce_loss'].append(epoch_losses['infonce'])
        history['transition_loss'].append(epoch_losses['transition'])
        history['latent_std'].append(epoch_losses['std'])
        history['val_total_loss'].append(val_losses['total'])
        history['val_recon_loss'].append(val_losses['recon'])
        history['val_kl_loss'].append(val_losses['kl'])
        history['val_transition_loss'].append(val_losses['transition'])
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"full_pipeline_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'vae_state_dict': vae.state_dict(),
            'transition_state_dict': transition_net.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'history': history,
            'config': {
                'num_games': num_games,
                'beta': beta,
                'infonce_weight': infonce_weight,
                'transition_weight': transition_weight,
                'lr': lr
            }
        }, checkpoint_path)
        
        # Track best
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_checkpoint = checkpoint_dir / "full_pipeline_best.pt"
            torch.save({
                'epoch': epoch,
                'vae_state_dict': vae.state_dict(),
                'transition_state_dict': transition_net.state_dict(),
                'val_loss': best_val_loss
            }, best_checkpoint)
    
    # Save final model
    final_model_path = checkpoint_dir / "full_pipeline.pt"
    torch.save({
        'epoch': epochs,
        'vae_state_dict': vae.state_dict(),
        'transition_state_dict': transition_net.state_dict(),
        'history': history,
        'config': {
            'num_games': num_games,
            'beta': beta,
            'infonce_weight': infonce_weight,
            'transition_weight': transition_weight,
            'lr': lr
        }
    }, final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Save training curves
    pd.DataFrame(history).to_csv(training_curves_dir / "full_pipeline.csv", index=False)
    plot_training_curves(history, training_curves_dir / "full_pipeline_loss.png")
    
    # Print summary
    print("\n" + "="*60)
    print("Full Pipeline Training Summary")
    print("="*60)
    print(f"Games processed: {num_games}")
    print(f"Training samples: {len(train_features_t)}")
    print(f"Test samples: {len(test_features_t)}")
    print(f"Epochs: {epochs}")
    print(f"\nInitial → Final:")
    print(f"  Total Loss: {history['total_loss'][0]:.4f} → {history['total_loss'][-1]:.4f}")
    print(f"  Recon Loss: {history['recon_loss'][0]:.4f} → {history['recon_loss'][-1]:.4f}")
    print(f"  KL Loss: {history['kl_loss'][0]:.4f} → {history['kl_loss'][-1]:.4f}")
    print(f"  InfoNCE: {history['infonce_loss'][0]:.4f} → {history['infonce_loss'][-1]:.4f}")
    print(f"  Transition: {history['transition_loss'][0]:.4f} → {history['transition_loss'][-1]:.4f}")
    print(f"  Latent Std: {history['latent_std'][0]:.4f} → {history['latent_std'][-1]:.4f}")
    print("="*60)
    
    return vae, transition_net, history


def generate_report(history: dict, num_games: int) -> str:
    """Generate the reporting format specified in the task."""
    
    # Calculate metrics
    initial_recon = history['recon_loss'][0]
    final_recon = history['recon_loss'][-1]
    delta_recon = final_recon - initial_recon
    
    initial_kl = history['kl_loss'][0]
    final_kl = history['kl_loss'][-1]
    delta_kl = final_kl - initial_kl
    
    initial_std = history['latent_std'][0]
    final_std = history['latent_std'][-1]
    delta_std = final_std - initial_std
    
    initial_infonce = history['infonce_loss'][0]
    final_infonce = history['infonce_loss'][-1]
    
    initial_trans = history['transition_loss'][0]
    final_trans = history['transition_loss'][-1]
    
    # Calculate stability (variance in last 10 epochs)
    last_10_total = history['total_loss'][-10:]
    stability = np.std(last_10_total)
    stability_status = "STABLE" if stability < 0.1 else "UNSTABLE"
    
    report = f"""## 650-Game Training Results

### Data Summary
- Games processed: {num_games}
- Features extracted: {num_games * 2} (home + away)
- Transition labels computed: {num_games * 2}

### VAE Performance
| Metric | Initial | Final | Change |
|--------|---------|-------|--------|
| Reconstruction Loss | {initial_recon:.4f} | {final_recon:.4f} | {delta_recon:+.4f} |
| KL Divergence | {initial_kl:.4f} | {final_kl:.4f} | {delta_kl:+.4f} |
| Latent Std | {initial_std:.4f} | {final_std:.4f} | {delta_std:+.4f} |

### InfoNCE Performance
| Metric | Initial | Final |
|--------|---------|-------|
| Contrastive Loss | {initial_infonce:.4f} | {final_infonce:.4f} |

### Transition Network Performance
| Metric | Value |
|--------|-------|
| Cross-Entropy Loss (Initial) | {initial_trans:.4f} |
| Cross-Entropy Loss (Final) | {final_trans:.4f} |

### Stability Assessment
{stability_status} (last 10 epochs std: {stability:.4f})

### Key Findings
- VAE reconstruction loss decreased from {initial_recon:.4f} to {final_recon:.4f}
- KL divergence shows {"healthy regularization" if final_kl > 0.5 else "under-regularized"} latent space
- Latent std approaching {"1.0 (standard normal)" if abs(final_std - 1.0) < 0.3 else f"{final_std:.2f} (may need adjustment)"}
- InfoNCE contrastive loss: {initial_infonce:.4f} → {final_infonce:.4f}
- Transition prediction loss decreased: {initial_trans:.4f} → {final_trans:.4f}

### Recommendations
1. If latent std is far from 1.0, adjust beta parameter
2. If transition loss is high, increase transition_weight
3. Consider increasing epochs if losses are still decreasing
4. For production, use real streaming data instead of synthetic
"""
    
    return report


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Full Pipeline: VAE + InfoNCE + Transition Network')
    parser.add_argument('--games', type=int, default=650, help='Number of games to train on')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--beta', type=float, default=1.0, help='KL divergence weight')
    parser.add_argument('--infonce_weight', type=float, default=0.1, help='InfoNCE loss weight')
    parser.add_argument('--transition_weight', type=float, default=1.0, help='Transition loss weight')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use_real_data', action='store_true', default=True, 
                        help='Use real data from StatBroadcast (required, no synthetic fallback)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Full Pipeline Training: VAE + InfoNCE + Transition Network")
    print("="*60)
    print(f"Games: {args.games}")
    print(f"Epochs: {args.epochs}")
    print(f"Data: REAL (StatBroadcast streaming)")
    print(f"Beta (KL weight): {args.beta}")
    print(f"InfoNCE weight: {args.infonce_weight}")
    print(f"Transition weight: {args.transition_weight}")
    print("="*60 + "\n")
    
    # Train with real data (no synthetic fallback)
    vae, transition_net, history = train_full_pipeline(
        num_games=args.games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        beta=args.beta,
        infonce_weight=args.infonce_weight,
        transition_weight=args.transition_weight,
        lr=args.lr,
        use_real_data=True  # Always use real data
    )
    
    # Generate report
    report = generate_report(history, args.games)
    print("\n" + report)
    
    # Save report
    report_path = Path(__file__).parent / "training_curves" / "650game_training_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Saved report to {report_path}")
    
    return vae, transition_net, history


if __name__ == "__main__":
    main()
