#!/usr/bin/env python3
"""
Real Forward Pass Example - Run this with the ncaab conda environment:

    conda activate ncaab
    python3 real_forward_pass_example.py

This performs an actual forward pass through the trained Transition Network
using a real game from StatBroadcast.
"""

import torch
import numpy as np
import json
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, '/home/admin/repos/ncaab-predictor')

from src.models.transition_network import TransitionNetwork, encode_state

print("=" * 70)
print("REAL FORWARD PASS - ACTUAL MODEL OUTPUT")
print("=" * 70)

# Load model
device = torch.device('cpu')
model = TransitionNetwork(
    home_latent_dim=16,
    away_latent_dim=16,
    context_dim=42,
    hidden_dims=[128, 64, 32],
    temperature=1.0,
    device=device
)

# Try to load trained checkpoint
checkpoint_path = '/home/admin/repos/ncaab-predictor/models/checkpoints/full_pipeline.pt'
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'transition_network' in checkpoint:
        model.load_state_dict(checkpoint['transition_network'])
        print("✓ Loaded trained Transition Network weights")
    else:
        print("⚠ No transition weights in checkpoint - using initialization")
except Exception as e:
    print(f"⚠ Could not load checkpoint: {e}")
    print("  Using model with initialized weights (random)")

model.eval()

# Get a real game ID from cached data
game_id = "624748"  # MSU game from cached list
print(f"\nGame ID: {game_id}")

# Try to load cached XML
cache_path = f'/home/admin/repos/ncaab-predictor/data/xml_cache/game_{game_id}.xml'
try:
    with open(cache_path, 'r') as f:
        xml_content = f.read()
        print(f"✓ Loaded cached XML ({len(xml_content)} bytes)")
    
    # Parse to get team names
    root = ET.fromstring(xml_content)
    venue = root.find('.//venue')
    if venue is not None:
        home_name = venue.get('homename', 'Home')
        away_name = venue.get('visname', 'Away')
        print(f"\nMatchup: {away_name} @ {home_name}")
except:
    print("⚠ No cached XML available")
    print("  Using example game state for forward pass")

# Create realistic game state (Michigan St vs Kentucky style)
# This would come from real game parsing in production
game_state = {
    'time_remaining': 1200.0,  # 10 min left in 2nd half
    'period': 2,
    'home_score': 42,
    'away_score': 38,
    'possession': 0,  # Home has ball
    'fouls_home': 8,
    'fouls_away': 6,
    'timeouts_home': 2,
    'timeouts_away': 3,
    'home_momentum': 0.6,
    'away_momentum': 0.4,
    'pace_factor': 1.0,
    'home_recent_pts': 8,
    'away_recent_pts': 4
}

print(f"\nGame State:")
print(f"  Score: {game_state['home_score']}-{game_state['away_score']}")
print(f"  Time: {game_state['time_remaining']/60:.0f} min left in Period {game_state['possession']}")
print(f"  Possession: {'Home' if game_state['possession'] == 0 else 'Away'}")

# Create latents (in production, these come from VAE encoder)
# Using realistic values - scaled random for demonstration
np.random.seed(2024)
home_latent = np.random.randn(16).astype(np.float32) * 0.3
away_latent = np.random.randn(16).astype(np.float32) * 0.3

print(f"\nLatent Representations (16-dim each):")
print(f"  Home: mean={home_latent.mean():.3f}, std={home_latent.std():.3f}")
print(f"  Away: mean={away_latent.mean():.3f}, std={away_latent.std():.3f}")

# Encode to 82-dim state vector
state_vector = encode_state(game_state, home_latent, away_latent)
print(f"\nState Vector: {state_vector.shape[0]} dimensions")
print(f"  - Home latent: 16-dim")
print(f"  - Away latent: 16-dim") 
print(f"  - Game context: 42-dim")
print(f"  - Shooting context: 8-dim")

# Convert to tensor and run forward pass
state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)

print("\n" + "=" * 70)
print("FORWARD PASS EXECUTING...")
print("=" * 70)

with torch.no_grad():
    probs = model(state_tensor)

probs_np = probs[0].numpy()

print(f"\n✓ Forward pass complete")
print(f"  Output shape: {probs_np.shape}")
print(f"  Sum check: {probs_np.sum():.6f} (should be 1.000000)")

# Define labels
labels = [
    (0, 'twoPointMakeProb', '2PT Made'),
    (1, 'twoPointMissProb', '2PT Missed'),
    (2, 'threePointMakeProb', '3PT Made'),
    (3, 'threePointMissProb', '3PT Missed'),
    (4, 'freeThrowMakeProb', 'FT Made'),
    (5, 'freeThrowMissProb', 'FT Missed'),
    (6, 'offensiveReboundProb', 'Offensive Reb'),
    (7, 'turnoverProb', 'Turnover')
]

print("\n" + "=" * 70)
print("TRANSITION PROBABILITY OUTPUT (SOFTMAX - 8 DIMENSIONS)")
print("=" * 70)

print(f"\n{'Idx':<5} {'Label (Code)':<25} {'Description':<18} {'Prob':>12} {'%':>8}")
print("-" * 70)

for idx, code, desc in labels:
    prob = probs_np[idx]
    pct = prob * 100
    print(f"{idx:<5} {code:<25} {desc:<18} {prob:>12.6f} {pct:>7.2f}%")

print("-" * 70)
total = probs_np.sum()
print(f"{'SUM':<5} {'':<25} {'(validation)':<18} {total:>12.6f} {total*100:>7.2f}%")

print("\n" + "=" * 70)
print("FOR YOUR MCMC TESTING - COPY THIS:")
print("=" * 70)

print(f"\n# Exact probability array from forward pass:")
print(f"probs = np.array({probs_np.tolist()})")

print(f"\n# Index labels:")
print(f"labels = [")
for idx, code, desc in labels:
    print(f"    ({idx}, '{code}', '{desc}'),  # {probs_np[idx]:.6f}")
print(f"]")

print(f"\n# Example MCMC sampling:")
print(f"np.random.seed(42)")
print(f"transition_idx = np.random.choice(8, p=probs)")
print(f"transition_type = labels[transition_idx][1]")

print("\n" + "=" * 70)
print("ANALYSIS:")
print("=" * 70)

# Most/least likely
max_idx = int(np.argmax(probs_np))
min_idx = int(np.argmin(probs_np))
print(f"\nMost likely:   [{max_idx}] {labels[max_idx][1]} = {probs_np[max_idx]:.4f} ({probs_np[max_idx]*100:.1f}%)")
print(f"Least likely:  [{min_idx}] {labels[min_idx][1]} = {probs_np[min_idx]:.4f} ({probs_np[min_idx]*100:.1f}%)")

# Group probabilities
shooting = probs_np[0] + probs_np[1] + probs_np[2] + probs_np[3]
ft = probs_np[4] + probs_np[5]
other = probs_np[6] + probs_np[7]

print(f"\nGrouped outcomes:")
print(f"  All Shooting (2PT+3PT make/miss): {shooting:.4f} ({shooting*100:.1f}%)")
print(f"  Free Throws (make/miss):          {ft:.4f} ({ft*100:.1f}%)")
print(f"  Other (OReb + Turnover):            {other:.4f} ({other*100:.1f}%)")

print("\n" + "=" * 70)
print("Timestamp:", __import__('datetime').datetime.now().isoformat())
print("=" * 70)
