"""
Test script to run forward pass on sample game 16960 (Michigan St. vs Kentucky).
"""

import sys
import os
import xml.etree.ElementTree as ET
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from src.models.transition_network import (
    TransitionNetwork,
    encode_state,
    sample_next_state,
    TRANSITION_NAMES
)
from src.data.xml_parser import parse_xml, extract_game_metadata


def parse_game_state_from_xml(xml_path: str) -> dict:
    """Parse game state from XML file."""
    with open(xml_path, 'r') as f:
        xml_content = f.read()
    
    root = ET.fromstring(xml_content)
    
    # Extract game metadata
    game_id = root.find('.//venue').attrib.get('gameid', 'unknown')
    
    # Extract game status
    status = root.find('.//status')
    period = int(status.attrib.get('period', 1))
    clock = status.attrib.get('clock', '20:00')
    
    # Parse clock to seconds
    def clock_to_seconds(clock_str: str) -> float:
        try:
            parts = clock_str.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        except:
            return 1200.0
    
    time_remaining = clock_to_seconds(clock)
    
    # Extract team info
    teams = {}
    for team in root.findall('.//team'):
        vh = team.attrib.get('vh', '')
        name = team.attrib.get('name', 'Unknown')
        
        # Get score
        linescore = team.find('.//linescore')
        if linescore is not None:
            score = int(linescore.attrib.get('score', 0))
        else:
            score = 0
        
        # Get totals
        totals = team.find('.//totals')
        if totals is not None:
            stats = totals.find('.//stats')
            if stats is not None:
                fouls = int(stats.attrib.get('pf', 0))
                turnovers = int(stats.attrib.get('to', 0))
                oreb = int(stats.attrib.get('oreb', 0))
                dreb = int(stats.attrib.get('dreb', 0))
            else:
                fouls, turnovers, oreb, dreb = 0, 0, 0, 0
        else:
            fouls, turnovers, oreb, dreb = 0, 0, 0, 0
        
        team_key = 'home' if vh == 'H' else 'away'
        teams[team_key] = {
            'name': name,
            'score': score,
            'fouls': fouls,
            'turnovers': turnovers,
            'oreb': oreb,
            'dreb': dreb
        }
    
    # Get possession info
    status_attrib = status.attrib
    poss = status_attrib.get('poss', 'V')  # V = visiting team has it
    
    return {
        'game_id': game_id,
        'time_remaining': time_remaining,
        'period': period,
        'home_score': teams.get('home', {}).get('score', 0),
        'away_score': teams.get('away', {}).get('score', 0),
        'home_fouls': teams.get('home', {}).get('fouls', 0),
        'away_fouls': teams.get('away', {}).get('fouls', 0),
        'home_turnovers': teams.get('away', {}).get('turnovers', 0),  # Turnovers caused
        'away_turnovers': teams.get('home', {}).get('turnovers', 0),
        'home_off_reb': teams.get('home', {}).get('oreb', 0),
        'away_off_reb': teams.get('away', {}).get('oreb', 0),
        'possession': 0 if poss == 'H' else 1,  # 0=home, 1=away
    }


def main():
    print("=" * 70)
    print("Transition Network Forward Pass Test")
    print("Game: Michigan St. vs Kentucky (Game ID: 16960)")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    model = TransitionNetwork(
        home_latent_dim=16,
        away_latent_dim=16,
        context_dim=42,
        hidden_dims=[128, 64, 32],
        temperature=1.0,
        device=device
    )
    model.eval()
    
    print(f"\nModel loaded successfully")
    print(f"  Temperature: {model.temperature}")
    print(f"  Input dim: 74 (home:16 + away:16 + context:42)")
    print(f"  Hidden dims: {model.hidden_dims}")
    print(f"  Output dim: 8")
    
    # Parse game state from XML
    xml_path = os.path.join(
        os.path.dirname(__file__),
        'tests', 'fixtures', 'sample_game_16960.xml'
    )
    
    # If not found, try absolute path
    if not os.path.exists(xml_path):
        xml_path = '/home/admin/repos/ncaab-predictor/tests/fixtures/sample_game_16960.xml'
    
    print(f"\nParsing game XML: {xml_path}")
    game_state = parse_game_state_from_xml(xml_path)
    
    print(f"\nGame State:")
    print(f"  Game ID: {game_state['game_id']}")
    print(f"  Period: {game_state['period']}")
    print(f"  Time Remaining: {game_state['time_remaining']:.0f}s")
    print(f"  Score: {game_state['home_score']} (Home) - {game_state['away_score']} (Away)")
    print(f"  Possession: {'Home' if game_state['possession'] == 0 else 'Away'}")
    print(f"  Fouls: Home={game_state['home_fouls']}, Away={game_state['away_fouls']}")
    
    # Create latent representations (in real use, these come from VAE encoder)
    # Using random latents for demonstration
    np.random.seed(42)
    home_latent = np.random.randn(16).astype(np.float32) * 0.5
    away_latent = np.random.randn(16).astype(np.float32) * 0.5
    
    # Encode game state to 74-dim vector
    state_vector = encode_state(game_state, home_latent, away_latent)
    
    print(f"\nState Vector:")
    print(f"  Total dimensions: {len(state_vector)}")
    print(f"  Home latent (0-15): mean={state_vector[:16].mean():.4f}")
    print(f"  Away latent (16-31): mean={state_vector[16:32].mean():.4f}")
    print(f"  Context (32-73): mean={state_vector[32:].mean():.4f}")
    
    # Run forward pass
    state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
    
    with torch.no_grad():
        probs = model(state_tensor)
    
    probs_np = probs[0].cpu().numpy()
    
    print("\n" + "=" * 70)
    print("TRANSITION PROBABILITIES")
    print("=" * 70)
    print(f"\nGame ID used: {game_state['game_id']}")
    print(f"\n8-Dimensional Transition Probabilities:")
    print("-" * 40)
    
    for i, name in TRANSITION_NAMES.items():
        print(f"  P({name:12s}): {probs_np[i]:.6f}")
    
    print("-" * 40)
    print(f"  Sum: {probs_np.sum():.6f}")
    
    # Verify it's a valid probability distribution
    assert abs(probs_np.sum() - 1.0) < 1e-5, "Probabilities should sum to 1!"
    assert (probs_np >= 0).all(), "All probabilities should be non-negative!"
    assert (probs_np <= 1).all(), "All probabilities should be <= 1!"
    
    print("\n✓ Valid probability distribution verified!")
    
    # Test sample_next_state
    print("\n" + "=" * 70)
    print("Testing sample_next_state function")
    print("=" * 70)
    
    next_state = sample_next_state(game_state, probs_np, home_latent, away_latent)
    
    print(f"\nCurrent State:")
    print(f"  Time: {game_state['time_remaining']:.0f}s, Period: {game_state['period']}")
    print(f"  Score: {game_state['home_score']}-{game_state['away_score']}")
    print(f"  Possession: {'Home' if game_state['possession'] == 0 else 'Away'}")
    
    print(f"\nSampled Next State:")
    print(f"  Time: {next_state['time_remaining']:.0f}s, Period: {next_state['period']}")
    print(f"  Score: {next_state['home_score']}-{next_state['away_score']}")
    print(f"  Possession: {'Home' if next_state['possession'] == 0 else 'Away'}")
    print(f"  Last transition: {next_state.get('last_transition', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    
    return probs_np, game_state['game_id']


if __name__ == "__main__":
    probs, game_id = main()
