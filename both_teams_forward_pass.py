#!/usr/bin/env python3
"""
Both Teams Forward Pass - Game 624748 (MSU vs Kentucky)

This shows the 8-dim transition probabilities for BOTH teams.
The probabilities change based on who has possession.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/admin/repos/ncaab-predictor')

from src.models.transition_network import TransitionNetwork, encode_state

print("=" * 70)
print("BOTH TEAMS FORWARD PASS - GAME 624748")
print("Michigan St (Home) vs Kentucky (Away)")
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

checkpoint_path = '/home/admin/repos/ncaab-predictor/models/checkpoints/full_pipeline.pt'
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'transition_network' in checkpoint:
        model.load_state_dict(checkpoint['transition_network'])
        print("✓ Loaded trained model")
except:
    print("⚠ Using initialized weights")

model.eval()

# Same game state, 10 min left in 2nd half
game_state = {
    'time_remaining': 600.0,
    'period': 2,
    'home_score': 42,
    'away_score': 38,
    'possession': 0,
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

# Team latents (same for both scenarios)
np.random.seed(2024)
home_latent = np.random.randn(16).astype(np.float32) * 0.3
away_latent = np.random.randn(16).astype(np.float32) * 0.3

print(f"\nScore: Michigan St 42 - Kentucky 38")
print(f"Time: 10:00 left in 2nd half")

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

# ============== MICHIGAN ST (HOME) WITH BALL ==============
print("\n" + "=" * 70)
print("SCENARIO 1: MICHIGAN ST (HOME) WITH POSSESSION")
print("=" * 70)

game_state['possession'] = 0  # Home has ball
state_home = encode_state(game_state, home_latent, away_latent)
state_tensor_home = torch.FloatTensor(state_home).unsqueeze(0)

with torch.no_grad():
    probs_home = model(state_tensor_home)[0].numpy()

print(f"\n{'Idx':<5} {'Label':<25} {'Prob':>12} {'%':>8}")
print("-" * 70)
for idx, code, desc in labels:
    prob = probs_home[idx]
    pct = prob * 100
    print(f"{idx:<5} {code:<25} {prob:>12.6f} {pct:>7.2f}%")
print("-" * 70)
print(f"{'SUM':<5} {'':<25} {probs_home.sum():>12.6f} {probs_home.sum()*100:>7.2f}%")

print(f"\n# For MCMC - Michigan St possession:")
print(f"probs_msu = np.array({[round(p, 6) for p in probs_home.tolist()]})")

# ============== KENTUCKY (AWAY) WITH BALL ==============
print("\n" + "=" * 70)
print("SCENARIO 2: KENTUCKY (AWAY) WITH POSSESSION")
print("=" * 70)

game_state['possession'] = 1  # Away has ball
state_away = encode_state(game_state, home_latent, away_latent)
state_tensor_away = torch.FloatTensor(state_away).unsqueeze(0)

with torch.no_grad():
    probs_away = model(state_tensor_away)[0].numpy()

print(f"\n{'Idx':<5} {'Label':<25} {'Prob':>12} {'%':>8}")
print("-" * 70)
for idx, code, desc in labels:
    prob = probs_away[idx]
    pct = prob * 100
    print(f"{idx:<5} {code:<25} {prob:>12.6f} {pct:>7.2f}%")
print("-" * 70)
print(f"{'SUM':<5} {'':<25} {probs_away.sum():>12.6f} {probs_away.sum()*100:>7.2f}%")

print(f"\n# For MCMC - Kentucky possession:")
print(f"probs_uk = np.array({[round(p, 6) for p in probs_away.tolist()]})")

# ============== COMPARISON ==============
print("\n" + "=" * 70)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 70)

print(f"\n{'Idx':<5} {'Label':<20} {'MSU (Home)':>12} {'UK (Away)':>12} {'Diff':>10}")
print("-" * 70)

for idx, code, desc in labels:
    msu_prob = probs_home[idx]
    uk_prob = probs_away[idx]
    diff = uk_prob - msu_prob
    diff_pct = diff * 100
    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
    print(f"{idx:<5} {code:<20} {msu_prob:>12.4f} {uk_prob:>12.4f} {diff_pct:>+9.2f}% {arrow}")

print("-" * 70)
print(f"\nKey Differences:")
max_diff_idx = np.argmax(np.abs(probs_away - probs_home))
max_diff = abs(probs_away[max_diff_idx] - probs_home[max_diff_idx]) * 100
print(f"  Largest difference: [{max_diff_idx}] {labels[max_diff_idx][1]} ({max_diff:+.2f}%)")

# Most likely for each
msu_max = np.argmax(probs_home)
uk_max = np.argmax(probs_away)
print(f"\nMost likely outcome:")
print(f"  MSU with ball: [{msu_max}] {labels[msu_max][1]} ({probs_home[msu_max]*100:.1f}%)")
print(f"  UK with ball:  [{uk_max}] {labels[uk_max][1]} ({probs_away[uk_max]*100:.1f}%)")

print("\n" + "=" * 70)
print("FOR MCMC - USE BOTH ARRAYS AND FLIP BASED ON POSSESSION:")
print("=" * 70)
print(f"""
if possession == 0:  # Home (MSU) has ball
    probs = np.array({[round(p, 6) for p in probs_home.tolist()]})
else:  # Away (UK) has ball
    probs = np.array({[round(p, 6) for p in probs_away.tolist()]})

transition_idx = np.random.choice(8, p=probs)

# After transition, flip possession (unless it's OReb which keeps it)
if transition_idx != 6:  # Not offensive rebound
    possession = 1 - possession  # Flip 0<->1
""")

print("=" * 70)
