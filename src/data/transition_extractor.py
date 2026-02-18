"""
Transition Extractor for NCAAB Prediction System.

Extracts 8-dimensional transition probabilities from game XML for both home and away teams.
Stores results in database via update functions.
"""

import logging
from typing import Tuple, Optional
import numpy as np

from src.data.playbyplay_parser import (
    parse_play_by_play,
    compute_transition_probabilities,
    validate_transition_probs
)

# Configure logging
logger = logging.getLogger(__name__)


def extract_transition_probabilities(
    xml_content: str,
    team_id: str,
    vh: str
) -> np.ndarray:
    """Extract 8-dim transition probabilities for a team from XML.
    
    Args:
        xml_content: Raw XML content as string
        team_id: Team identifier (e.g., 'MSU', 'KEN')
        vh: 'H' for home team, 'V' for away team
        
    Returns:
        8-dimensional numpy array of transition probabilities
        [twoPointMake, twoPointMiss, threePointMake, threePointMiss,
         freeThrowMake, freeThrowMiss, offensiveRebound, turnover]
    """
    try:
        probs = compute_transition_probabilities(xml_content, team_id, vh)
        
        # Validate
        if not validate_transition_probs(probs):
            logger.warning(f"Invalid transition probs for {team_id}, using uniform")
            return np.ones(8, dtype=np.float32) / 8.0
        
        return probs
        
    except Exception as e:
        logger.error(f"Error extracting transition probs for {team_id}: {e}")
        # Return uniform on error
        return np.ones(8, dtype=np.float32) / 8.0


def extract_game_transitions(
    xml_content: str,
    home_team_id: str,
    away_team_id: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract transition probabilities for both teams in a game.
    
    Args:
        xml_content: Raw XML content as string
        home_team_id: Home team identifier
        away_team_id: Away team identifier
        
    Returns:
        Tuple of (home_probs, away_probs) as 8-dim numpy arrays
    """
    # Extract for home team
    home_probs = extract_transition_probabilities(xml_content, home_team_id, 'H')
    
    # Extract for away team
    away_probs = extract_transition_probabilities(xml_content, away_team_id, 'V')
    
    logger.info(f"Extracted transition probs: home={home_probs.round(3)}, away={away_probs.round(3)}")
    
    return home_probs, away_probs


def update_game_transitions(
    game_id: str,
    xml_content: str,
    home_team_id: str,
    away_team_id: str
) -> bool:
    """Extract and store transition probabilities for a game.
    
    Args:
        game_id: The game ID
        xml_content: Raw XML content
        home_team_id: Home team identifier
        away_team_id: Away team identifier
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract probabilities
        home_probs, away_probs = extract_game_transitions(
            xml_content, home_team_id, away_team_id
        )
        
        # Store in database
        from src.data.database import store_transition_probs
        
        store_transition_probs(game_id, home_probs.tolist(), away_probs.tolist())
        
        logger.info(f"Stored transition probs for game {game_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update game transitions for {game_id}: {e}")
        return False


def get_team_transition_probs(
    game_id: str,
    team: str = 'home'
) -> Optional[np.ndarray]:
    """Get transition probabilities for a team from a specific game.
    
    Args:
        game_id: The game ID
        team: 'home' or 'away'
        
    Returns:
        8-dim numpy array or None if not found
    """
    try:
        from src.data.database import fetch_transition_probs
        
        result = fetch_transition_probs(game_id)
        
        if result is None:
            return None
        
        home_probs, away_probs = result
        
        if team == 'home':
            return np.array(home_probs, dtype=np.float32)
        else:
            return np.array(away_probs, dtype=np.float32)
            
    except Exception as e:
        logger.error(f"Failed to get transition probs for {team} in {game_id}: {e}")
        return None


# Batch processing utilities
def process_games_for_transitions(
    game_ids: list,
    loader
) -> dict:
    """Process multiple games to extract transition probabilities.
    
    Args:
        game_ids: List of game IDs to process
        loader: StreamingXMLoader or similar to fetch game XML
        
    Returns:
        Dictionary with results: {game_id: {'success': bool, 'home_probs': array, 'away_probs': array}}
    """
    results = {}
    
    for game_id in game_ids:
        try:
            # Fetch XML
            xml_content = loader.fetch_game_xml(game_id)
            
            if xml_content is None:
                results[game_id] = {'success': False, 'error': 'No XML content'}
                continue
            
            # Try to get team IDs from database
            from src.data.database import get_all_stored_game_ids
            
            games = get_all_stored_game_ids()
            game_info = next((g for g in games if g['game_id'] == game_id), None)
            
            if game_info is None:
                results[game_id] = {'success': False, 'error': 'Game not in database'}
                continue
            
            home_team_id = game_info.get('home_team_id', 'HOME')
            away_team_id = game_info.get('away_team_id', 'AWAY')
            
            # Extract probabilities
            home_probs, away_probs = extract_game_transitions(
                xml_content, home_team_id, away_team_id
            )
            
            # Store in database
            store_transition_probs(game_id, home_probs.tolist(), away_probs.tolist())
            
            results[game_id] = {
                'success': True,
                'home_probs': home_probs,
                'away_probs': away_probs
            }
            
        except Exception as e:
            results[game_id] = {'success': False, 'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Test extraction
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
    
    print("Testing transition extraction...")
    
    home_probs, away_probs = extract_game_transitions(sample_xml, "MSU", "KEN")
    
    print(f"Home (MSU): {home_probs.round(4)}")
    print(f"Away (KEN): {away_probs.round(4)}")
    
    print("\nExtraction test complete!")
