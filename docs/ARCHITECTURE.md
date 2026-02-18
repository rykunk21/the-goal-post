# NCAAB Prediction Architecture

## Two-Head Training System

### Head 1: VAE (Representation Learning)
- **Input**: 80-dim team statistical features
- **Encoder**: 80 → [128,64] → 16-dim (μ, logvar)
- **Decoder**: 16-dim → [64,128] → 80-dim reconstruction
- **Loss**: L_recon + β × L_KL
  - Reconstruction: Binary Cross Entropy
  - KL: -0.5 × Σ(1 + logvar - μ² - exp(logvar))
- **Purpose**: Learn compressed, meaningful team representations

### Head 2: InfoNCE (Contrastive Learning)
- **Input**: 16-dim latents from VAE encoder
- **Positive pairs**: Teams with similar 8-dim transition probability vectors
- **Negative pairs**: Teams with dissimilar transition patterns
- **Loss**: -log[exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ)]
  - sim: Cosine similarity
  - τ: Temperature (0.07)
- **Purpose**: Force latents to be predictive of game dynamics

### Head 3: Transition Network (Prediction)
- **Input**: 74-dim state (home_latent:16 + away_latent:16 + context:42)
- **Hidden**: [128, 64, 32]
- **Output**: 8-dim transition probabilities
  - twoPointMake, twoPointMiss, threePointMake, threePointMiss
  - freeThrowMake, freeThrowMiss, offensiveRebound, turnover
- **Loss**: Cross-entropy against ground truth transition counts
- **Purpose**: Predict game state transitions from team representations

## Training Flow

1. **Data Streaming** (1 req/sec rate limit)
   - Load game ID from list
   - Fetch XML: `http://archive.statbroadcast.com/{game_id}.xml`
   - Extract 80-dim features for home/away teams
   - Extract 8-dim transition probabilities from play-by-play

2. **Forward Pass**
   - VAE: features → latents (μ, logvar) → reconstruction
   - InfoNCE: latents compared via transition similarity
   - Transition NN: (latent_home, latent_away, context) → pred_transitions

3. **Loss Computation**
   - L_total = L_recon + β×L_KL + λ×L_infonce + γ×L_transition
   - Default: β=1.0, λ=0.1, γ=1.0

4. **Backpropagation**
   - VAE params: updated by recon + KL + (indirectly by InfoNCE)
   - InfoNCE: updates encoder to produce better similarity structure
   - Transition NN: learns to predict from frozen (or joint) latents
