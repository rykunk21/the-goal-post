#!/usr/bin/env python3
"""
Quick test of streaming loader with cached game IDs.
Tests anti-blocking measures with archive URLs.
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.streaming_loader import StreamingXMLoader

# Load cached game IDs
with open('data/statbroadcast_game_ids.json', 'r') as f:
    game_ids_dict = json.load(f)

# Flatten all game IDs
all_game_ids = []
for team, ids in game_ids_dict.items():
    all_game_ids.extend(ids)

unique_game_ids = list(set(all_game_ids))
print(f"Loaded {len(unique_game_ids)} unique game IDs from cache")

# Test streaming with a small sample
test_ids = unique_game_ids[:20]  # Test first 20 games
print(f"Testing with {len(test_ids)} game IDs")

loader = StreamingXMLoader()

success_count = 0
fail_count = 0
error_403 = 0

for i, game_id in enumerate(test_ids):
    print(f"\nTesting game {i+1}/{len(test_ids)}: {game_id}")
    try:
        home, away = loader.fetch_game_features(int(game_id))
        if home is not None and away is not None:
            print(f"  ✓ Success! Features shape: {home.shape}")
            success_count += 1
        else:
            print(f"  ✗ Failed (no data)")
            fail_count += 1
    except Exception as e:
        error_msg = str(e)
        if '403' in error_msg:
            print(f"  ✗ 403 Forbidden")
            error_403 += 1
        else:
            print(f"  ✗ Error: {e}")
        fail_count += 1

loader.close()

print(f"\n=== Results ===")
print(f"Total: {len(test_ids)}")
print(f"Success: {success_count}")
print(f"Failed: {fail_count}")
print(f"403 errors: {error_403}")
