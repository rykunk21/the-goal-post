"""
Play-by-Play Parser for NCAAB Prediction System.

Parses StatBroadcast XML play-by-play data to extract possession outcomes.
Computes 8-dimensional transition probabilities matching Steve.js structure:
- 0: twoPointMakeProb
- 1: twoPointMissProb
- 2: threePointMakeProb
- 3: threePointMissProb
- 4: freeThrowMakeProb
- 5: freeThrowMissProb
- 6: offensiveReboundProb
- 7: turnoverProb
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


def parse_play_by_play(xml_content: str) -> List[Dict[str, Any]]:
    """Parse play-by-play from StatBroadcast XML.
    
    Args:
        xml_content: XML content as string
        
    Returns:
        List of play dictionaries with keys:
            - action: GOOD, MISS, REBOUND, TURNOVER, etc.
            - type: 3PTR, FT, LAYUP, JUMPER, DUNK, etc.
            - team: team id
            - vh: 'V' (visitor/away) or 'H' (home)
            - period: period number
            - time: time remaining in period
    """
    plays = []
    
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return plays
    
    # Find play-by-play section
    pbp = root.find('.//playbyplay')
    if pbp is None:
        # Try alternative structure
        pbp = root.find('.//pbp')
    if pbp is None:
        print("No play-by-play section found in XML")
        return plays
    
    # Parse each play
    for period_elem in pbp.findall('period'):
        period_num = int(period_elem.get('num', 1))
        
        for play in period_elem.findall('play'):
            action = play.get('action', '')
            play_type = play.get('type', '')
            team = play.get('team', '')
            vh = play.get('vh', '')  # V=visitor/away, H=home
            time = play.get('time', '0:00')
            
            # Only include relevant plays
            if action in ['GOOD', 'MISS', 'REBOUND', 'TURNOVER', 'STEAL']:
                plays.append({
                    'action': action,
                    'type': play_type,
                    'team': team,
                    'vh': vh,
                    'period': period_num,
                    'time': time
                })
    
    return plays


def categorize_play(play: Dict[str, Any]) -> Optional[str]:
    """Categorize a play into one of 8 Steve.js transition categories.
    
    Args:
        play: Play dictionary from parse_play_by_play
        
    Returns:
        Transition category string or None if not a scoring play
    """
    action = play.get('action', '')
    play_type = play.get('type', '')
    
    if action == 'MISS':
        # Missed shot
        if play_type == '3PTR':
            return 'threePointMiss'
        elif play_type in ['LAYUP', 'DUNK', 'JUMPER', 'HOOK', 'TIP']:
            return 'twoPointMiss'
        elif play_type == 'FT':
            return 'freeThrowMiss'
        else:
            # Default to 2PT for unknown types
            return 'twoPointMiss'
    
    elif action == 'GOOD':
        # Made shot
        if play_type == '3PTR':
            return 'threePointMake'
        elif play_type in ['LAYUP', 'DUNK', 'JUMPER', 'HOOK', 'TIP']:
            return 'twoPointMake'
        elif play_type == 'FT':
            return 'freeThrowMake'
        else:
            # Default to 2PT for unknown types
            return 'twoPointMake'
    
    elif action == 'REBOUND':
        # Rebound - determine if offensive or defensive
        # Offensive rebound = team that missed keeps possession
        # We need context to know this, but we'll mark as offensive 
        # if the next play is by the same team
        return 'offensiveRebound'  # Placeholder, needs context
    
    elif action in ['TURNOVER', 'STEAL']:
        return 'turnover'
    
    return None


def compute_transition_counts(
    plays: List[Dict[str, Any]],
    team_id: str,
    vh: str
) -> Dict[str, int]:
    """Count possession outcomes for a specific team.
    
    Args:
        plays: List of all plays
        team_id: Team identifier to filter for
        vh: 'H' for home team, 'V' for away team
        
    Returns:
        Dictionary with counts for each transition type
    """
    counts = {
        'twoPointMakes': 0,
        'twoPointMisses': 0,
        'threePointMakes': 0,
        'threePointMisses': 0,
        'freeThrowMakes': 0,
        'freeThrowMisses': 0,
        'offensiveRebounds': 0,
        'turnovers': 0,
    }
    
    # Get team's plays
    team_plays = [p for p in plays if p.get('vh') == vh]
    
    # Track previous play to determine rebound type
    prev_play = None
    
    for i, play in enumerate(team_plays):
        action = play.get('action', '')
        play_type = play.get('type', '')
        
        if action == 'GOOD':
            if play_type == '3PTR':
                counts['threePointMakes'] += 1
            elif play_type == 'FT':
                counts['freeThrowMakes'] += 1
            else:  # LAYUP, DUNK, JUMPER, etc.
                counts['twoPointMakes'] += 1
                
        elif action == 'MISS':
            if play_type == '3PTR':
                counts['threePointMisses'] += 1
            elif play_type == 'FT':
                counts['freeThrowMisses'] += 1
            else:
                counts['twoPointMisses'] += 1
                
        elif action == 'REBOUND':
            # Determine if offensive or defensive
            # Offensive rebound = after own missed shot
            # Defensive rebound = after opponent missed shot
            if prev_play and prev_play.get('vh') == vh:
                # Previous play was by same team, so this is defensive
                pass  # Don't count defensive rebounds
            else:
                # Previous play was by opponent, this is offensive
                counts['offensiveRebounds'] += 1
                
        elif action in ['TURNOVER', 'STEAL']:
            counts['turnovers'] += 1
        
        prev_play = play
    
    return counts


def compute_transition_probabilities(
    xml_content: str,
    team_id: str,
    vh: str
) -> np.ndarray:
    """Compute 8-dim transition probabilities for a team from XML.
    
    Args:
        xml_content: XML content as string
        team_id: Team identifier
        vh: 'H' for home team, 'V' for away team
        
    Returns:
        8-dimensional numpy array of transition probabilities
        [twoPointMake, twoPointMiss, threePointMake, threePointMiss,
         freeThrowMake, freeThrowMiss, offensiveRebound, turnover]
    """
    # Parse play-by-play
    plays = parse_play_by_play(xml_content)
    
    if not plays:
        # Return uniform distribution if no plays found
        return np.ones(8, dtype=np.float32) / 8.0
    
    # Compute counts for this team
    counts = compute_transition_counts(plays, team_id, vh)
    
    # Convert to array in Steve.js order
    count_array = np.array([
        counts['twoPointMakes'],
        counts['twoPointMisses'],
        counts['threePointMakes'],
        counts['threePointMisses'],
        counts['freeThrowMakes'],
        counts['freeThrowMisses'],
        counts['offensiveRebounds'],
        counts['turnovers'],
    ], dtype=np.float32)
    
    # Normalize to probabilities
    total = count_array.sum()
    
    if total == 0:
        # No events - return uniform
        return np.ones(8, dtype=np.float32) / 8.0
    
    probs = count_array / total
    
    # Validate
    assert np.isclose(probs.sum(), 1.0, atol=1e-5), \
        f"Probabilities must sum to 1.0, got {probs.sum()}"
    assert all(p >= 0 for p in probs), \
        "All probabilities must be non-negative"
    
    return probs


def validate_transition_probs(probs: np.ndarray) -> bool:
    """Validate transition probability vector.
    
    Args:
        probs: 8-dim probability vector
        
    Returns:
        True if valid, False otherwise
    """
    if len(probs) != 8:
        print(f"Error: Expected 8 probabilities, got {len(probs)}")
        return False
    
    if not np.isclose(probs.sum(), 1.0, atol=1e-5):
        print(f"Error: Probabilities sum to {probs.sum():.6f}, expected 1.0")
        return False
    
    if any(p < 0 for p in probs):
        print("Error: Negative probabilities found")
        return False
    
    return True


# Example usage and testing
if __name__ == "__main__":
    # Test with sample XML
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <game>
        <playbyplay>
            <period num="1">
                <play action="GOOD" type="LAYUP" team="MSU" vh="H" time="19:30"/>
                <play action="GOOD" type="3PTR" team="KEN" vh="V" time="19:00"/>
                <play action="MISS" type="LAYUP" team="MSU" vh="H" time="18:30"/>
                <play action="REBOUND" team="KEN" vh="V" time="18:25"/>
                <play action="GOOD" type="FT" team="KEN" vh="V" time="18:00"/>
                <play action="MISS" type="FT" team="MSU" vh="H" time="17:30"/>
                <play action="TURNOVER" team="KEN" vh="V" time="17:00"/>
                <play action="GOOD" type="JUMPER" team="MSU" vh="H" time="16:30"/>
            </period>
        </playbyplay>
    </game>
    """
    
    print("Testing play-by-play parser...")
    
    # Parse plays
    plays = parse_play_by_play(sample_xml)
    print(f"Parsed {len(plays)} plays:")
    for play in plays:
        print(f"  {play}")
    
    # Compute transition probabilities for home team (MSU)
    home_probs = compute_transition_probabilities(sample_xml, "MSU", "H")
    print(f"\nHome team (MSU) transition probabilities:")
    print(f"  twoPointMake:     {home_probs[0]:.4f}")
    print(f"  twoPointMiss:     {home_probs[1]:.4f}")
    print(f"  threePointMake:  {home_probs[2]:.4f}")
    print(f"  threePointMiss:  {home_probs[3]:.4f}")
    print(f"  freeThrowMake:   {home_probs[4]:.4f}")
    print(f"  freeThrowMiss:   {home_probs[5]:.4f}")
    print(f"  offensiveRebound:{home_probs[6]:.4f}")
    print(f"  turnover:        {home_probs[7]:.4f}")
    print(f"  Sum: {home_probs.sum():.4f}")
    
    # Validate
    is_valid = validate_transition_probs(home_probs)
    print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
