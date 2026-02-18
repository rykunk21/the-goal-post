"""
XML Parsing Functions for NCAAB Data Pipeline

Provides functions to parse XML game data and extract structured information.
"""

import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional


def parse_xml(raw_xml: str) -> dict:
    """
    Parse raw XML string into a dictionary representation.
    
    Args:
        raw_xml: String containing XML data
        
    Returns:
        Dictionary representation of XML with tags as keys
        
    Raises:
        ET.ParseError: If XML is malformed
    """
    if not raw_xml or not raw_xml.strip():
        raise ValueError("Empty XML string provided")
    
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as e:
        raise ET.ParseError(f"Failed to parse XML: {e}")
    
    def _element_to_dict(element: ET.Element) -> Any:
        """Recursively convert XML element to dict."""
        children = list(element)
        
        # If no children, return text content
        if not children:
            text = element.text.strip() if element.text else ""
            return text if text else None
        
        # Build dict from children
        result = {}
        for child in children:
            child_data = _element_to_dict(child)
            
            if child.tag in result:
                # Handle multiple elements with same tag (make it a list)
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        # Add attributes if present
        if element.attrib:
            result['@attributes'] = element.attrib
            
        return result
    
    return {root.tag: _element_to_dict(root)}


def extract_game_metadata(xml_dict: dict) -> dict:
    """
    Extract game metadata from parsed XML dictionary.
    
    Args:
        xml_dict: Dictionary returned from parse_xml()
        
    Returns:
        Dictionary containing game metadata (date, venue, teams, etc.)
    """
    if not xml_dict:
        return {}
    
    # Find the game node (could be at different levels)
    def find_node(d: dict, target_keys: set) -> Optional[dict]:
        """Find a node with any of the target keys."""
        if isinstance(d, dict):
            if any(k in d for k in target_keys):
                return d
            for v in d.values():
                result = find_node(v, target_keys)
                if result:
                    return result
        elif isinstance(d, list):
            for item in d:
                result = find_node(item, target_keys)
                if result:
                    return result
        return None
    
    # Common metadata keys
    metadata_keys = {'game_id', 'date', 'venue', 'start_time', 'season', 'conference'}
    
    game_node = find_node(xml_dict, metadata_keys)
    if not game_node:
        return {}
    
    # Extract known metadata fields
    metadata = {}
    known_fields = {
        'game_id': 'game_id',
        'date': 'date',
        'venue': 'venue',
        'start_time': 'start_time',
        'season': 'season',
        'conference': 'conference',
        'game_date': 'date',
    }
    
    for xml_key, out_key in known_fields.items():
        if xml_key in game_node:
            metadata[out_key] = game_node[xml_key]
    
    # Try to extract team info
    if 'home_team' in game_node:
        metadata['home_team'] = game_node['home_team']
    if 'away_team' in game_node:
        metadata['away_team'] = game_node['away_team']
    if 'teams' in game_node:
        metadata['teams'] = game_node['teams']
        
    # Extract final score if present
    if 'home_score' in game_node:
        metadata['home_score'] = game_node['home_score']
    if 'away_score' in game_node:
        metadata['away_score'] = game_node['away_score']
    
    return metadata


def extract_team_stats(xml_dict: dict, team_type: str) -> dict:
    """
    Extract statistics for a specific team from parsed XML.
    
    Args:
        xml_dict: Dictionary returned from parse_xml()
        team_type: 'home' or 'away' (or 'winning'/'losing' for some feeds)
        
    Returns:
        Dictionary containing team statistics
        
    Raises:
        ValueError: If team_type is not valid
    """
    if not xml_dict:
        return {}
    
    valid_team_types = {'home', 'away', 'winning', 'losing', 'team1', 'team2', 'home_team', 'away_team', 'team'}
    if team_type not in valid_team_types:
        raise ValueError(f"Invalid team_type: {team_type}. Must be one of {valid_team_types}")
    
    # Find team stats in the XML
    def find_team_stats(d: dict, ttype: str) -> Optional[dict]:
        """Find statistics for the specified team type."""
        if not isinstance(d, dict):
            return None
        
        # Direct key match
        if ttype in d:
            return d[ttype]
        
        # Check in teams array
        if 'teams' in d and isinstance(d['teams'], list):
            for team in d['teams']:
                if isinstance(team, dict):
                    if team.get('type') == ttype or team.get('@attributes', {}).get('type') == ttype:
                        return team
        
        # Check in boxscore or similar structures
        for key in ['boxscore', 'team_stats', 'statistics', 'stats']:
            if key in d:
                result = find_team_stats(d[key], ttype)
                if result:
                    return result
        
        # Recursive search
        for v in d.values():
            if isinstance(v, (dict, list)):
                result = find_team_stats(v, ttype)
                if result:
                    return result
        
        return None
    
    stats = find_team_stats(xml_dict, team_type)
    
    if not stats:
        return {}
    
    # Normalize common stat field names
    normalized = {}
    
    # Points
    if 'points' in stats:
        normalized['points'] = stats['points']
    elif 'score' in stats:
        normalized['points'] = stats['score']
    
    # Field goals
    fg_fields = {'fgm': 'fgm', 'fgm Made': 'fgm', 'field_goals_made': 'fgm',
                 'fga': 'fga', 'fg attempts': 'fga', 'field_goals_attempted': 'fga'}
    for k, v in fg_fields.items():
        if k in stats:
            normalized[v] = stats[k]
    
    # Three-pointers
    three_fields = {'fg3m': 'fg3m', '3pm': 'fg3m', 'three_pm': 'fg3m',
                    'fg3a': 'fg3a', '3pa': 'fg3a', 'three_pa': 'fg3a'}
    for k, v in three_fields.items():
        if k in stats:
            normalized[v] = stats[k]
    
    # Free throws
    ft_fields = {'ftm': 'ftm', 'ftm Made': 'ftm', 'free_throws_made': 'ftm',
                 'fta': 'fta', 'ft attempts': 'fta', 'free_throws_attempted': 'fta'}
    for k, v in ft_fields.items():
        if k in stats:
            normalized[v] = stats[k]
    
    # Rebounds
    reb_fields = {'rebounds': 'rebounds', 'reb': 'rebounds', 'total_reb': 'rebounds',
                  'oreb': 'oreb', 'off_reb': 'oreb', 'offensive_reb': 'oreb',
                  'dreb': 'dreb', 'def_reb': 'dreb', 'defensive_reb': 'dreb'}
    for k, v in reb_fields.items():
        if k in stats:
            normalized[v] = stats[k]
    
    # Turnovers
    to_fields = {'turnovers': 'turnovers', 'to': 'turnovers', 'tov': 'turnovers'}
    for k, v in to_fields.items():
        if k in stats:
            normalized[v] = stats[k]
    
    # Assists
    ast_fields = {'assists': 'assists', 'ast': 'assists'}
    for k, v in ast_fields.items():
        if k in stats:
            normalized[v] = stats[k]
    
    # Steals
    stl_fields = {'steals': 'steals', 'stl': 'steals'}
    for k, v in stl_fields.items():
        if k in stats:
            normalized[v] = stats[k]
    
    # Blocks
    blk_fields = {'blocks': 'blocks', 'blk': 'blocks', 'block': 'blocks'}
    for k, v in blk_fields.items():
        if k in stats:
            normalized[v] = stats[k]
    
    # Personal fouls
    pf_fields = {'fouls': 'fouls', 'pf': 'fouls', 'personal_fouls': 'fouls'}
    for k, v in pf_fields.items():
        if k in stats:
            normalized[v] = stats[k]
    
    # Possessions (often computed or in advanced stats)
    if 'possessions' in stats:
        normalized['possessions'] = stats['possessions']
    elif 'pace' in stats:
        normalized['possessions'] = stats['pace']
    
    # Add team type identifier
    normalized['team_type'] = team_type
    
    return normalized
