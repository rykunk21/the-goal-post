"""
Test utility helpers for NCAAB predictor tests.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional, Union


def load_json_fixture(fixture_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON fixture file."""
    with open(fixture_path, "r") as f:
        return json.load(f)


def load_xml_fixture(fixture_path: Union[str, Path]) -> ET.Element:
    """Load and parse an XML fixture file."""
    tree = ET.parse(fixture_path)
    return tree.getroot()


def parse_game_xml(xml_string: str) -> Dict[str, Any]:
    """
    Parse StatBroadcast game XML into a dictionary.
    
    Args:
        xml_string: Raw XML string
        
    Returns:
        Dictionary with parsed game data
    """
    root = ET.fromstring(xml_string)
    
    game_data = {
        "game_id": root.findtext("gameId"),
        "date": root.findtext("date"),
        "time": root.findtext("time"),
        "period": root.findtext("period"),
        "clock": root.findtext("clock"),
        "status": root.findtext("status"),
    }
    
    # Parse venue
    venue = root.find("venue")
    if venue is not None:
        game_data["venue"] = {
            "name": venue.findtext("name"),
            "city": venue.findtext("city"),
            "state": venue.findtext("state"),
        }
    
    # Parse home team
    home_team = root.find("homeTeam")
    if home_team is not None:
        game_data["home_team"] = {
            "team_id": home_team.findtext("teamId"),
            "name": home_team.findtext("name"),
            "conference": home_team.findtext("conference"),
            "score": int(home_team.findtext("score", "0")),
            "period_score": int(home_team.findtext("periodScore", "0")),
        }
        
        stats = home_team.find("statistics")
        if stats is not None:
            game_data["home_team"]["statistics"] = _parse_team_stats(stats)
    
    # Parse away team
    away_team = root.find("awayTeam")
    if away_team is not None:
        game_data["away_team"] = {
            "team_id": away_team.findtext("teamId"),
            "name": away_team.findtext("name"),
            "conference": away_team.findtext("conference"),
            "score": int(away_team.findtext("score", "0")),
            "period_score": int(away_team.findtext("periodScore", "0")),
        }
        
        stats = away_team.find("statistics")
        if stats is not None:
            game_data["away_team"]["statistics"] = _parse_team_stats(stats)
    
    return game_data


def _parse_team_stats(stats_elem: ET.Element) -> Dict[str, Any]:
    """Parse team statistics element into a dictionary."""
    stats = {}
    for child in stats_elem:
        tag = child.tag
        text = child.text
        if text is not None and text.isdigit():
            stats[tag] = int(text)
        elif text is not None:
            try:
                stats[tag] = float(text)
            except ValueError:
                stats[tag] = text
    return stats


def create_mock_game_data(
    game_id: str = "2024021501",
    home_team: str = "Duke",
    away_team: str = "North Carolina",
    home_score: int = 78,
    away_score: int = 72,
    **overrides: Any
) -> Dict[str, Any]:
    """
    Create mock game data for testing.
    
    Args:
        game_id: Game identifier
        home_team: Home team name
        away_team: Away team name
        home_score: Home team score
        away_score: Away team score
        **overrides: Additional fields to override
        
    Returns:
        Mock game data dictionary
    """
    base_data = {
        "game_id": game_id,
        "date": "2024-02-15",
        "time": "19:00:00",
        "venue": {
            "name": "Cameron Indoor Stadium",
            "city": "Durham",
            "state": "NC",
        },
        "home_team": {
            "team_id": "duke",
            "name": home_team,
            "conference": "ACC",
            "score": home_score,
            "period_score": home_score // 2,
            "statistics": {
                "fieldGoalsMade": 28,
                "fieldGoalsAttempted": 55,
                "threePointersMade": 9,
                "threePointersAttempted": 22,
                "freeThrowsMade": 13,
                "freeThrowsAttempted": 18,
                "reboundsOffensive": 10,
                "reboundsDefensive": 22,
                "reboundsTotal": 32,
                "assists": 15,
                "turnovers": 8,
                "steals": 6,
                "blocks": 3,
                "fouls": 12,
            },
        },
        "away_team": {
            "team_id": "unc",
            "name": away_team,
            "conference": "ACC",
            "score": away_score,
            "period_score": away_score // 2,
            "statistics": {
                "fieldGoalsMade": 26,
                "fieldGoalsAttempted": 58,
                "threePointersMade": 8,
                "threePointersAttempted": 21,
                "freeThrowsMade": 12,
                "freeThrowsAttempted": 16,
                "reboundsOffensive": 7,
                "reboundsDefensive": 25,
                "reboundsTotal": 32,
                "assists": 12,
                "turnovers": 12,
                "steals": 5,
                "blocks": 2,
                "fouls": 15,
            },
        },
        "period": "2",
        "clock": "0:32",
        "status": "final",
    }
    
    base_data.update(overrides)
    return base_data


def compare_game_data(
    actual: Dict[str, Any],
    expected: Dict[str, Any],
    ignore_keys: Optional[list] = None
) -> tuple[bool, Optional[str]]:
    """
    Compare two game data dictionaries.
    
    Args:
        actual: Actual game data
        expected: Expected game data
        ignore_keys: Keys to ignore in comparison
        
    Returns:
        Tuple of (is_equal, error_message)
    """
    ignore_keys = ignore_keys or []
    
    for key, expected_value in expected.items():
        if key in ignore_keys:
            continue
            
        if key not in actual:
            return False, f"Missing key: {key}"
        
        actual_value = actual[key]
        
        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            is_equal, error = compare_game_data(actual_value, expected_value, ignore_keys)
            if not is_equal:
                return False, f"{key}: {error}"
        elif actual_value != expected_value:
            return False, f"Key '{key}': expected {expected_value}, got {actual_value}"
    
    return True, None
