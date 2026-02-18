"""
Transition Probability Network for NCAAB Prediction System.

Predicts transition probabilities (8-dim) from game state embeddings.
Uses VAE latent representations + game context to predict next-state transitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Dict, Any, List
import numpy as np


# Transition type indices (Steve.js correct 8-dim structure)
TRANSITION_TYPES = {
    'twoPointMake': 0,    # P(2PT made)
    'twoPointMiss': 1,    # P(2PT missed)
    'threePointMake': 2,  # P(3PT made)
    'threePointMiss': 3,  # P(3PT missed)
    'freeThrowMake': 4,   # P(FT made)
    'freeThrowMiss': 5,   # P(FT missed)
    'offensiveRebound': 6,  # P(offensive rebound - possession retained)
    'turnover': 7,        # P(turnover - possession lost)
}

# Reverse mapping
TRANSITION_NAMES = {v: k for k, v in TRANSITION_TYPES.items()}


class TransitionNetwork(nn.Module):
    """Transition Probability Network for basketball game state prediction.
    
    Architecture:
        Input: 82-dim state vector
            - Home team latent: 16-dim (from VAE encoder)
            - Away team latent: 16-dim (from VAE encoder)
            - Shooting context: 8-dim (home team transition probabilities)
            - Game context: 42-dim (time, score, possession, etc.)
        Hidden: [128, 64, 32]
        Output: 8-dim transition probabilities with temperature scaling
        
    Features:
        - Temperature scaling (tau=1.0 default) for calibration
        - ReLU activations
        - Softmax for valid probability distribution
        - Proper device placement
    """
    
    def __init__(
        self,
        home_latent_dim: int = 16,
        away_latent_dim: int = 16,
        shooting_context_dim: int = 8,
        context_dim: int = 42,
        hidden_dims: List[int] = [128, 64, 32],
        temperature: float = 1.0,
        device: Optional[torch.device] = None
    ):
        super(TransitionNetwork, self).__init__()
        
        self.home_latent_dim = home_latent_dim
        self.away_latent_dim = away_latent_dim
        self.shooting_context_dim = shooting_context_dim
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.temperature = temperature
        
        # Input dimension = home + away + shooting_context + context
        input_dim = home_latent_dim + away_latent_dim + shooting_context_dim + context_dim
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Build hidden layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, 8)
        
        # Initialize weights
        self._init_weights()
        
        # Move to device
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        """Forward pass through transition network.
        
        Args:
            state_vector: Input tensor of shape (batch_size, 82)
                Contains [home_latent(16), away_latent(16), shooting_context(8), context(42)]
                
        Returns:
            Transition probabilities of shape (batch_size, 8)
            Each row sums to 1.0 (valid probability distribution)
        """
        # Hidden layers with ReLU
        hidden = self.hidden_layers(state_vector)
        
        # Output logits
        logits = self.output_layer(hidden)
        
        # Apply temperature scaling before softmax
        scaled_logits = logits / self.temperature
        
        # Apply softmax for valid probability distribution
        probs = F.softmax(scaled_logits, dim=-1)
        
        return probs
    
    def predict_transitions(
        self,
        state_vector: torch.Tensor,
        latents_home: Optional[torch.Tensor] = None,
        latents_away: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict transition probabilities.
        
        Convenience method that assembles state vector from components.
        
        Args:
            state_vector: Pre-assembled 82-dim state vector
            latents_home: Optional 16-dim home latents (for verification)
            latents_away: Optional 16-dim away latents (for verification)
            
        Returns:
            Transition probabilities of shape (batch_size, 8)
        """
        return self.forward(state_vector)
    
    def set_temperature(self, temperature: float):
        """Update temperature parameter for calibration.
        
        Args:
            temperature: New temperature value (lower = sharper, higher = softer)
        """
        self.temperature = temperature


def encode_state(
    game_state: Dict[str, Any],
    home_latent: Optional[np.ndarray] = None,
    away_latent: Optional[np.ndarray] = None,
    home_shooting_probs: Optional[np.ndarray] = None,
    away_shooting_probs: Optional[np.ndarray] = None
) -> np.ndarray:
    """Encode game state to 82-dimensional state vector.
    
    Args:
        game_state: Dictionary containing game context information
            Expected keys:
                - time_remaining: float (seconds left in game)
                - home_score: int
                - away_score: int
                - possession: int (0=home, 1=away)
                - period: int (1-4 or 5 for OT)
                - fouls_home: int
                - fouls_away: int
                - timeouts_home: int
                - timeouts_away: int
                - ... any other game context
        home_latent: Optional 16-dim numpy array for home team latent
        away_latent: Optional 16-dim numpy array for away team latent
        home_shooting_probs: Optional 8-dim numpy array for home team's shooting transition probs
        away_shooting_probs: Optional 8-dim numpy array for away team's shooting transition probs
        
    Returns:
        82-dimensional state vector as numpy array
        [home_latent(16), away_latent(16), home_shooting_probs(8), context(42)]
    """
    # Default latents to zeros if not provided
    if home_latent is None:
        home_latent = np.zeros(16, dtype=np.float32)
    if away_latent is None:
        away_latent = np.zeros(16, dtype=np.float32)
    
    # Default shooting probs to uniform if not provided
    # (equal probability for each of 8 outcomes)
    if home_shooting_probs is None:
        home_shooting_probs = np.ones(8, dtype=np.float32) / 8.0
    if away_shooting_probs is None:
        away_shooting_probs = np.ones(8, dtype=np.float32) / 8.0
    
    # Build context vector (42-dim)
    context = np.zeros(42, dtype=np.float32)
    
    # Time-related features (0-3)
    time_remaining = game_state.get('time_remaining', 1200.0)  # 20 min = 1200 sec
    context[0] = time_remaining / 1200.0  # Normalized time remaining
    
    period = game_state.get('period', 1)
    context[1] = period / 5.0  # Normalized period (max 5 with OT)
    
    # Is overtime
    context[2] = 1.0 if period > 4 else 0.0
    
    # Time in current period
    context[3] = game_state.get('time_in_period', 0.0) / 600.0  # 10 min = 600 sec
    
    # Score features (4-9)
    home_score = game_state.get('home_score', 0)
    away_score = game_state.get('away_score', 0)
    context[4] = home_score / 100.0  # Normalized home score
    context[5] = away_score / 100.0  # Normalized away score
    context[6] = (home_score - away_score) / 20.0  # Score differential
    context[7] = abs(home_score - away_score) / 20.0  # Absolute score differential
    context[8] = 1.0 if home_score > away_score else 0.0  # Home leading
    context[9] = 1.0 if home_score == away_score else 0.0  # Game tied
    
    # Possession features (10-13)
    possession = game_state.get('possession', 0)  # 0=home, 1=away
    context[10] = 1.0 if possession == 0 else 0.0  # Home has ball
    context[11] = 1.0 if possession == 1 else 0.0  # Away has ball
    
    # Recent scoring (last N possessions)
    context[12] = game_state.get('home_recent_pts', 0) / 10.0  # Recent home scoring
    context[13] = game_state.get('away_recent_pts', 0) / 10.0  # Recent away scoring
    
    # Foul features (14-17)
    fouls_home = game_state.get('fouls_home', 0)
    fouls_away = game_state.get('fouls_away', 0)
    context[14] = fouls_home / 10.0  # Normalized home fouls
    context[15] = fouls_away / 10.0  # Normalized away fouls
    context[16] = 1.0 if fouls_home >= 5 else 0.0  # Home in bonus
    context[17] = 1.0 if fouls_away >= 5 else 0.0  # Away in bonus
    
    # Timeout features (18-21)
    timeouts_home = game_state.get('timeouts_home', 3)
    timeouts_away = game_state.get('timeouts_away', 3)
    context[18] = timeouts_home / 3.0  # Normalized home timeouts
    context[19] = timeouts_away / 3.0  # Normalized away timeouts
    context[20] = 1.0 if timeouts_home == 0 else 0.0  # Home out of timeouts
    context[21] = 1.0 if timeouts_away == 0 else 0.0  # Away out of timeouts
    
    # Momentum/pace features (22-25)
    context[22] = game_state.get('home_momentum', 0.5)  # Home team momentum
    context[23] = game_state.get('away_momentum', 0.5)  # Away team momentum
    context[24] = game_state.get('pace_factor', 1.0)  # Pace factor
    context[25] = game_state.get('game_flow', 0.5)  # Game flow indicator
    
    # Game situation features (26-31)
    # Close game indicator (within 5 points)
    score_diff = abs(home_score - away_score)
    context[26] = 1.0 if score_diff <= 5 else 0.0  # Close game
    context[27] = 1.0 if score_diff <= 3 else 0.0  # Very close game
    context[28] = 1.0 if score_diff >= 15 else 0.0  # Blowout
    context[29] = 1.0 if time_remaining < 120 and score_diff <= 5 else 0.0  # Crunch time
    context[30] = 1.0 if time_remaining < 60 else 0.0  # Final minute
    context[31] = 1.0 if period == 2 and time_remaining > 300 else 0.0  # Early/mid game
    
    # Starters/bench features (32-35)
    context[32] = game_state.get('home_starters_fouls', 0) / 5.0
    context[33] = game_state.get('away_starters_fouls', 0) / 5.0
    context[34] = game_state.get('home_foul_trouble', 0.0)  # Players with 3+ fouls
    context[35] = game_state.get('away_foul_trouble', 0.0)
    
    # Rebound/turnover situation (36-39)
    context[36] = game_state.get('home_off_reb', 0.0) / 5.0  # Offensive rebounds
    context[37] = game_state.get('away_off_reb', 0.0) / 5.0
    context[38] = game_state.get('home_turnovers', 0.0) / 10.0  # Turnovers
    context[39] = game_state.get('away_turnovers', 0.0) / 10.0
    
    # Free throw situation (40-41)
    context[40] = game_state.get('ft_attempts_diff', 0.0) / 10.0  # FT disparity
    context[41] = game_state.get('in_bound_play', 0.0)  # In-bounding ball
    
    # Assemble full state vector: [home_latent(16), away_latent(16), shooting_context(8), context(42)]
    # shooting_context uses home team's shooting probabilities for the team with possession
    # If home has possession, use home_shooting_probs; otherwise use away_shooting_probs
    possession = game_state.get('possession', 0)
    shooting_context = home_shooting_probs if possession == 0 else away_shooting_probs
    
    state_vector = np.concatenate([home_latent, away_latent, shooting_context, context])
    
    return state_vector


def sample_next_state(
    current_state: Dict[str, Any],
    transition_probs: np.ndarray,
    home_latent: Optional[np.ndarray] = None,
    away_latent: Optional[np.ndarray] = None,
    home_shooting_probs: Optional[np.ndarray] = None,
    away_shooting_probs: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Sample next game state based on 8-dim shooting transition probabilities.
    
    The 8 transition probabilities are (Steve.js structure):
    - 0: twoPointMakeProb - 2PT shot made
    - 1: twoPointMissProb - 2PT shot missed
    - 2: threePointMakeProb - 3PT shot made  
    - 3: threePointMissProb - 3PT shot missed
    - 4: freeThrowMakeProb - Free throw made
    - 5: freeThrowMissProb - Free throw missed
    - 6: offensiveReboundProb - Offensive rebound (possession retained)
    - 7: turnoverProb - Turnover (possession lost)
    
    Args:
        current_state: Current game state dictionary
        transition_probs: 8-dim probability vector (sums to 1.0)
        home_latent: Home team latent representation (16-dim) - unused but kept for API
        away_latent: Away team latent representation (16-dim) - unused but kept for API
        home_shooting_probs: Home team's 8-dim shooting probabilities
        away_shooting_probs: Away team's 8-dim shooting probabilities
        
    Returns:
        Next state dictionary with updated game information
    """
    # Validate probability sum
    prob_sum = transition_probs.sum()
    if not np.isclose(prob_sum, 1.0, atol=1e-5):
        # Normalize if slightly off
        transition_probs = transition_probs / prob_sum
    
    # Sample transition type based on probabilities
    transition_idx = np.random.choice(8, p=transition_probs)
    transition_type = TRANSITION_NAMES[transition_idx]
    
    # Initialize next state as copy of current
    next_state = current_state.copy()
    
    # Get current game state
    time_remaining = current_state.get('time_remaining', 1200.0)
    period = current_state.get('period', 1)
    home_score = current_state.get('home_score', 0)
    away_score = current_state.get('away_score', 0)
    possession = current_state.get('possession', 0)  # 0=home, 1=away
    
    # Default time decrement (average possession time ~20 seconds)
    time_decrement = 20.0
    
    # Determine which team is shooting based on possession
    shooting_team = 0 if possession == 0 else 1  # 0=home, 1=away
    
    # Handle each transition type (Steve.js shooting outcomes)
    if transition_type == 'twoPointMake':
        # 2PT made - 2 points, possession changes
        points = 2
        if shooting_team == 0:
            home_score += points
        else:
            away_score += points
        possession = 1 - possession  # Ball changes possession
        
    elif transition_type == 'twoPointMiss':
        # 2PT missed - check for offensive rebound (handled by transition 6)
        # For now, assume defense gets ball
        possession = 1 - possession
        
    elif transition_type == 'threePointMake':
        # 3PT made - 3 points, possession changes
        points = 3
        if shooting_team == 0:
            home_score += points
        else:
            away_score += points
        possession = 1 - possession
        
    elif transition_type == 'threePointMiss':
        # 3PT missed - check for offensive rebound (handled by transition 6)
        # For now, assume defense gets ball
        possession = 1 - possession
        
    elif transition_type == 'freeThrowMake':
        # Free throw made - 1 point, possession usually stays
        points = 1
        if shooting_team == 0:
            home_score += points
        else:
            away_score += points
        # FT typically doesn't change possession (except and-1)
        # For simplicity, keep possession the same
        # NOTE: Could extend to handle "and-1" situation
        
    elif transition_type == 'freeThrowMiss':
        # Free throw missed - could lead to rebound
        # For simplicity, assume defense gets ball
        possession = 1 - possession
        
    elif transition_type == 'offensiveRebound':
        # Offensive rebound - possession retained!
        # This is a special case - the team keeps possession
        # No time decrement for offensive rebound (usually quick)
        time_decrement = 5.0  # Quick tip-in attempt
        # possession stays the same (team retains)
        
    elif transition_type == 'turnover':
        # Turnover - possession lost!
        possession = 1 - possession
        # Turnover can happen anywhere, slightly less time
        time_decrement = 15.0
    
    # Update time
    new_time = max(0.0, time_remaining - time_decrement)
    
    # Check for period end
    if new_time <= 0 and period < 5:
        period += 1
        new_time = 600.0  # Reset to 10 minutes (or 5 for OT)
    
    # Update state
    next_state['home_score'] = home_score
    next_state['away_score'] = away_score
    next_state['possession'] = possession
    next_state['time_remaining'] = new_time
    next_state['period'] = period
    next_state['last_transition'] = transition_type
    
    # Validate output sums to 1.0
    new_sum = sum([
        transition_probs[0],  # twoPointMake
        transition_probs[1],  # twoPointMiss
        transition_probs[2],  # threePointMake
        transition_probs[3],  # threePointMiss
        transition_probs[4],  # freeThrowMake
        transition_probs[5],  # freeThrowMiss
        transition_probs[6],  # offensiveRebound
        transition_probs[7],  # turnover
    ])
    assert np.isclose(new_sum, 1.0, atol=1e-5), f"Transition probs must sum to 1.0, got {new_sum}"
    
    return next_state


class TransitionNetworkTrainer:
    """Training utilities for TransitionNetwork with InfoNCE contrastive learning."""
    
    def __init__(
        self,
        model: TransitionNetwork,
        learning_rate: float = 1e-3,
        temperature: float = 0.07,
        device: Optional[torch.device] = None
    ):
        self.model = model
        
        if device is None:
            self.device = model.device
        else:
            self.device = device
            model.to(device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # InfoNCE temperature for contrastive learning
        self.infonce_temperature = temperature
        self.infonce_loss = InfoNCETrainingLoss(temperature)
    
    def train_step(
        self,
        state_batch: torch.Tensor,
        true_transitions: Optional[torch.Tensor] = None,
        use_infonce: bool = True
    ) -> Dict[str, float]:
        """Single training step.
        
        Args:
            state_batch: Batch of state vectors (batch_size, 74)
            true_transitions: Optional true transition labels (batch_size,)
            use_infonce: Whether to use InfoNCE contrastive loss
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        # Forward pass
        pred_probs = self.model(state_batch)
        
        if true_transitions is not None:
            # Cross-entropy loss against true transitions
            ce_loss = F.cross_entropy(
                torch.log(pred_probs + 1e-8),
                true_transitions.to(self.device)
            )
            loss = ce_loss
        else:
            # Use InfoNCE if no labels
            loss = torch.tensor(0.0, device=self.device)
        
        # InfoNCE contrastive loss on latent representations
        if use_infonce and true_transitions is None:
            # Extract latents from state vector
            home_latents = state_batch[:, :16]
            away_latents = state_batch[:, 16:32]
            
            # Compute contrastive loss
            infonce = self.infonce_loss(home_latents, away_latents)
            loss = loss + infonce
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'ce_loss': ce_loss.item() if true_transitions is not None else 0.0
        }
    
    def evaluate(
        self,
        state_batch: torch.Tensor,
        true_transitions: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Evaluate on validation data.
        
        Args:
            state_batch: Batch of state vectors
            true_transitions: Optional true transition labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            pred_probs = self.model(state_batch)
            
            metrics = {}
            
            if true_transitions is not None:
                ce_loss = F.cross_entropy(
                    torch.log(pred_probs + 1e-8),
                    true_transitions.to(self.device)
                )
                metrics['ce_loss'] = ce_loss.item()
                
                # Accuracy
                preds = pred_probs.argmax(dim=-1)
                accuracy = (preds == true_transitions.to(self.device)).float().mean()
                metrics['accuracy'] = accuracy.item()
            
            # Entropy of predictions (uncertainty)
            entropy = -(pred_probs * torch.log(pred_probs + 1e-8)).sum(dim=-1).mean()
            metrics['entropy'] = entropy.item()
            
            return metrics


class InfoNCETrainingLoss(nn.Module):
    """InfoNCE Loss for training transition network with contrastive learning."""
    
    def __init__(self, temperature: float = 0.07):
        super(InfoNCETrainingLoss, self).__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings_i: torch.Tensor,
        embeddings_j: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute InfoNCE loss.
        
        Args:
            embeddings_i: First set of embeddings (batch_size, dim)
            embeddings_j: Second set of embeddings (batch_size, dim)
            labels: Optional class labels for supervised contrastive learning
            
        Returns:
            Contrastive loss scalar
        """
        # L2 normalize
        embeddings_i = F.normalize(embeddings_i, p=2, dim=1)
        embeddings_j = F.normalize(embeddings_j, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings_i, embeddings_j.t) / self.temperature
        
        batch_size = embeddings_i.size(0)
        
        if labels is None:
            # Unsupervised: diagonal are positive pairs
            labels = torch.arange(batch_size, device=embeddings_i.device)
            loss = F.cross_entropy(similarity, labels)
        else:
            # Supervised: mask positive pairs by label
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.t).float()
            mask = mask - torch.eye(batch_size, device=embeddings_i.device)
            
            exp_sim = torch.exp(similarity)
            denom = exp_sim.sum(dim=1, keepdim=True)
            pos_sim = (exp_sim * mask).sum(dim=1)
            num_pos = mask.sum(dim=1).clamp(min=1)
            
            loss = -torch.log(pos_sim / denom.squeeze()).mean()
        
        return loss


# Forward pass example and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Transition Network Forward Pass Example")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    model = TransitionNetwork(
        home_latent_dim=16,
        away_latent_dim=16,
        shooting_context_dim=8,
        context_dim=42,
        hidden_dims=[128, 64, 32],
        temperature=1.0,
        device=device
    )
    
    print(f"\nModel architecture:")
    print(f"  Input: 82-dim (home:16 + away:16 + shooting:8 + context:42)")
    print(f"  Hidden: [128, 64, 32]")
    print(f"  Output: 8-dim (softmax with temperature=1.0)")
    
    # Create random input (batch of 4)
    batch_size = 4
    state_vector = torch.randn(batch_size, 82, device=device)
    
    print(f"\nInput shape: {state_vector.shape}")
    
    # Forward pass
    probs = model(state_vector)
    
    print(f"\nOutput shape: {probs.shape}")
    print(f"Probabilities sum to 1: {probs.sum(dim=-1).allclose(torch.ones(batch_size, device=device))}")
    
    # Show sample probabilities (Steve.js structure)
    print(f"\nSample transition probabilities (Steve.js 8-dim structure):")
    print(f"  [0] twoPointMake:    {probs[0, 0].item():.4f}")
    print(f"  [1] twoPointMiss:    {probs[0, 1].item():.4f}")
    print(f"  [2] threePointMake:  {probs[0, 2].item():.4f}")
    print(f"  [3] threePointMiss:  {probs[0, 3].item():.4f}")
    print(f"  [4] freeThrowMake:   {probs[0, 4].item():.4f}")
    print(f"  [5] freeThrowMiss:   {probs[0, 5].item():.4f}")
    print(f"  [6] offensiveRebound:{probs[0, 6].item():.4f}")
    print(f"  [7] turnover:        {probs[0, 7].item():.4f}")
    print(f"  Sum: {probs[0].sum().item():.4f}")
    
    # Test encode_state with example data
    print("\n" + "=" * 60)
    print("Testing encode_state function")
    print("=" * 60)
    
    game_state = {
        'time_remaining': 600.0,  # 10 minutes left
        'period': 2,
        'home_score': 45,
        'away_score': 42,
        'possession': 0,  # Home has ball
        'fouls_home': 3,
        'fouls_away': 4,
        'timeouts_home': 2,
        'timeouts_away': 2,
    }
    
    # Use example latent vectors (normalized)
    home_latent = np.ones(16, dtype=np.float32) * 0.5
    away_latent = np.ones(16, dtype=np.float32) * 0.5
    home_shooting_probs = np.array([0.25, 0.15, 0.15, 0.15, 0.10, 0.05, 0.10, 0.05], dtype=np.float32)
    away_shooting_probs = np.array([0.20, 0.18, 0.12, 0.18, 0.12, 0.05, 0.10, 0.05], dtype=np.float32)
    
    state_vec = encode_state(game_state, home_latent, away_latent, home_shooting_probs, away_shooting_probs)
    print(f"\nEncoded state shape: {state_vec.shape}")
    print(f"  Home latent (0-15): {state_vec[:16].mean():.4f}")
    print(f"  Away latent (16-31): {state_vec[16:32].mean():.4f}")
    print(f"  Shooting context (32-39): {state_vec[32:40]}")
    print(f"  Context (40-81): {state_vec[40:].mean():.4f}")
    
    # Test forward pass with real state
    print("\n" + "=" * 60)
    print("Forward pass with encoded game state")
    print("=" * 60)
    
    state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
    probs = model(state_tensor)
    
    print(f"\nTransition probabilities (Steve.js structure):")
    for i, name in TRANSITION_NAMES.items():
        print(f"  P({name}): {probs[0, i].item():.4f}")
    
    # Test sample_next_state
    print("\n" + "=" * 60)
    print("Testing sample_next_state")
    print("=" * 60)
    
    probs_np = probs[0].cpu().numpy()
    next_state = sample_next_state(game_state, probs_np, home_latent, away_latent)
    
    print(f"\nCurrent state:")
    print(f"  Time: {game_state['time_remaining']:.0f}s, Period: {game_state['period']}")
    print(f"  Score: {game_state['home_score']}-{game_state['away_score']}")
    print(f"  Possession: {'Home' if game_state['possession'] == 0 else 'Away'}")
    
    print(f"\nNext state:")
    print(f"  Time: {next_state['time_remaining']:.0f}s, Period: {next_state['period']}")
    print(f"  Score: {next_state['home_score']}-{next_state['away_score']}")
    print(f"  Possession: {'Home' if next_state['possession'] == 0 else 'Away'}")
    print(f"  Last transition: {next_state.get('last_transition', 'N/A')}")
    
    # Test trainer
    print("\n" + "=" * 60)
    print("Testing TransitionNetworkTrainer")
    print("=" * 60)
    
    trainer = TransitionNetworkTrainer(
        model=model,
        learning_rate=1e-3,
        temperature=0.07,
        device=device
    )
    
    # Generate random training data (82-dim: 16+16+8+42)
    num_samples = 100
    train_states = torch.randn(num_samples, 82, device=device)
    train_labels = torch.randint(0, 8, (num_samples,))
    
    # Train one step
    metrics = trainer.train_step(train_states, train_labels, use_infonce=False)
    print(f"\nTraining step metrics: {metrics}")
    
    # Evaluate
    eval_metrics = trainer.evaluate(train_states[:10], train_labels[:10])
    print(f"Evaluation metrics: {eval_metrics}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
