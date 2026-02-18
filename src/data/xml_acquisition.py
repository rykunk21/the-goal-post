"""
XML data acquisition for NCAAB prediction system.
Handles fetching game data from StatBroadcast/ESPN APIs.
"""

import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

from .database import (
    is_game_id_stored,
    init_database,
    store_game_id_locally,
    get_all_stored_game_ids,
    update_team_from_game,
)

# Configuration
STATBROADCAST_BASE_URL = "https://api.statbroadcast.com/v1"
ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

# Default headers for requests
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/xml, text/xml, application/json",
}

# Cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "xml_cache"


def _ensure_cache_dir() -> None:
    """Ensure the XML cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ==================== XML Fetching ====================


def fetch_game_xml(game_id: str, use_cache: bool = True) -> Optional[str]:
    """
    Fetch XML data for a specific game ID.
    
    Args:
        game_id: The StatBroadcast game ID
        use_cache: Whether to use cached XML if available
    
    Returns:
        XML string or None if fetch fails
    """
    cache_file = CACHE_DIR / f"game_{game_id}.xml"
    
    # Check cache first
    if use_cache and cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Fetch from StatBroadcast API
    # Note: This is a placeholder - actual API endpoint needs to be configured
    url = f"{STATBROADCAST_BASE_URL}/games/{game_id}/boxscore"
    
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
        response.raise_for_status()
        
        xml_content = response.text
        
        # Cache the result
        _ensure_cache_dir()
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        return xml_content
    
    except requests.RequestException as e:
        print(f"Failed to fetch XML for game {game_id}: {e}")
        return None


def fetch_game_xml_from_file(game_id: str) -> Optional[str]:
    """
    Load XML from a local file (for testing).
    
    Args:
        game_id: The game ID
    
    Returns:
        XML string or None if file not found
    """
    cache_file = CACHE_DIR / f"game_{game_id}.xml"
    
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    return None


def save_xml_to_cache(game_id: str, xml_content: str) -> None:
    """
    Save XML content to cache.
    
    Args:
        game_id: The game ID
        xml_content: XML string to cache
    """
    _ensure_cache_dir()
    cache_file = CACHE_DIR / f"game_{game_id}.xml"
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(xml_content)


# ==================== Game ID Acquisition ====================


def fetch_game_ids_from_api(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_refresh: bool = False
) -> list[dict]:
    """
    Get list of all game IDs from StatBroadcast/ESPN.
    
    This function fetches game IDs for a date range and stores them locally.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (defaults to 30 days ago)
        end_date: End date in YYYY-MM-DD format (defaults to today)
        force_refresh: Force re-fetch even if games already stored
    
    Returns:
        List of game ID records with metadata
    """
    # Default to last 30 days
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    print(f"Fetching game IDs from {start_date} to {end_date}")
    
    game_ids = []
    
    # Try ESPN API first (more reliable for historical data)
    espn_games = _fetch_game_ids_from_espn(start_date, end_date)
    game_ids.extend(espn_games)
    
    # Try StatBroadcast API as fallback
    sb_games = _fetch_game_ids_from_statbroadcast(start_date, end_date)
    
    # Merge results, avoiding duplicates
    existing_ids = {g["game_id"] for g in game_ids}
    for game in sb_games:
        if game["game_id"] not in existing_ids:
            game_ids.append(game)
    
    # Store in database
    stored_count = 0
    for game in game_ids:
        if store_game_id_locally(
            game_id=game["game_id"],
            home_team_id=game.get("home_team_id"),
            away_team_id=game.get("away_team_id"),
            game_date=game.get("game_date"),
            sport="mens-college-basketball"
        ):
            stored_count += 1
    
    print(f"Found {len(game_ids)} games, stored {stored_count} new game IDs")
    
    return game_ids


def _fetch_game_ids_from_espn(start_date: str, end_date: str) -> list[dict]:
    """
    Fetch game IDs from ESPN API.
    
    Args:
        start_date: Start date in YYYY-MM-DD
        end_date: End date in YYYY-MM-DD
    
    Returns:
        List of game records
    """
    games = []
    
    # Convert dates to ISO format for ESPN
    start_iso = datetime.strptime(start_date, "%Y-%m-%d").isoformat()
    end_iso = datetime.strptime(end_date, "%Y-%m-%d").isoformat()
    
    url = f"{ESPN_BASE_URL}/scoreboard"
    params = {
        "dates": f"{start_date.replace('-', '')}-{end_date.replace('-', '')}",
        "limit": 1000,
    }
    
    try:
        response = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        for competition in data.get("events", []):
            game_id = competition.get("id")
            date_str = competition.get("date")
            
            # Extract teams
            home_team = None
            away_team = None
            
            for competitor in competition.get("competitions", [{}]).get("competitors", []):
                team_data = competitor.get("team", {})
                if competitor.get("homeAway") == "home":
                    home_team = team_data.get("abbreviation")
                else:
                    away_team = team_data.get("abbreviation")
            
            # Parse date
            game_date = None
            if date_str:
                game_date = datetime.fromisoformat(date_str.replace("Z", "+00:00")).strftime("%Y-%m-%d")
            
            games.append({
                "game_id": game_id,
                "home_team_id": home_team,
                "away_team_id": away_team,
                "game_date": game_date,
                "source": "espn",
            })
    
    except requests.RequestException as e:
        print(f"ESPN API error: {e}")
    except json.JSONDecodeError as e:
        print(f"ESPN JSON decode error: {e}")
    
    return games


def _fetch_game_ids_from_statbroadcast(start_date: str, end_date: str) -> list[dict]:
    """
    Fetch game IDs from StatBroadcast API.
    
    Note: This requires API key configuration. Placeholder implementation.
    
    Args:
        start_date: Start date in YYYY-MM-DD
        end_date: End date in YYYY-MM-DD
    
    Returns:
        List of game records
    """
    games = []
    
    # StatBroadcast API endpoint for game listings
    # This is a placeholder - actual implementation depends on API key
    url = f"{STATBROADCAST_BASE_URL}/games"
    params = {
        "sport": "mbb",  # Men's basketball
        "start_date": start_date,
        "end_date": end_date,
    }
    
    try:
        response = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        for game in data.get("games", []):
            games.append({
                "game_id": str(game.get("game_id")),
                "home_team_id": game.get("home", {}).get("abbreviation"),
                "away_team_id": game.get("away", {}).get("abbreviation"),
                "game_date": game.get("date"),
                "source": "statbroadcast",
            })
    
    except requests.RequestException:
        # Silently fail if StatBroadcast API not available
        pass
    
    return games


# ==================== Sample Data Functions ====================


def load_sample_game() -> Optional[str]:
    """
    Load the sample game XML for testing.
    
    Returns:
        XML string or None
    """
    sample_path = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "sample_game_16960.xml"
    
    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    return None


def parse_game_from_xml(xml_content: str) -> dict:
    """
    Parse game information from XML.
    
    Args:
        xml_content: XML string
    
    Returns:
        Dictionary with parsed game data
    """
    root = ET.fromstring(xml_content)
    
    # Get venue info
    venue = root.find("venue")
    
    game_data = {
        "game_id": venue.get("gameid") if venue is not None else None,
        "competition_id": venue.get("competitionid") if venue is not None else None,
        "statbroadcast_id": venue.get("sbid") if venue is not None else None,
        "date": venue.get("date") if venue is not None else None,
        "location": venue.get("location") if venue is not None else None,
        "home_team_id": venue.get("homeid") if venue is not None else None,
        "home_team_name": venue.get("homename") if venue is not None else None,
        "away_team_id": venue.get("visid") if venue is not None else None,
        "away_team_name": venue.get("visname") if venue is not None else None,
        "gender": venue.get("gender") if venue is not None else None,
    }
    
    # Get status
    status = root.find("status")
    if status is not None:
        game_data["status"] = {
            "complete": status.get("complete"),
            "period": status.get("period"),
            "clock": status.get("clock"),
            "gamestatus": status.get("gamestatus"),
        }
    
    # Get teams
    teams = root.findall("team")
    for team in teams:
        vh = team.get("vh")
        if vh == "H":
            game_data["home_team"] = _parse_team(team)
        elif vh == "V":
            game_data["away_team"] = _parse_team(team)
    
    return game_data


def _parse_team(team_elem: ET.Element) -> dict:
    """Parse team data from XML element."""
    team_id = team_elem.get("id")
    team_name = team_elem.get("name")
    
    # Get linescore
    linescore = team_elem.find("linescore")
    score = None
    if linescore is not None:
        score = linescore.get("score")
    
    # Get totals
    totals = team_elem.find("totals")
    stats = {}
    if totals is not None:
        stats_elem = totals.find("stats")
        if stats_elem is not None:
            stats = dict(stats_elem.attrib)
    
    return {
        "team_id": team_id,
        "team_name": team_name,
        "score": score,
        "stats": stats,
    }


def process_sample_game() -> dict:
    """
    Process the sample game and store in database.
    
    Returns:
        Parsed game data
    """
    xml_content = load_sample_game()
    if xml_content is None:
        raise ValueError("Sample game not found")
    
    game_data = parse_game_from_xml(xml_content)
    
    # Store game ID
    if game_data.get("game_id"):
        store_game_id_locally(
            game_id=game_data["game_id"],
            home_team_id=game_data.get("home_team_id"),
            away_team_id=game_data.get("away_team_id"),
            game_date=game_data.get("date"),
        )
    
    # Update team info
    if game_data.get("home_team_id"):
        update_team_from_game(
            team_id=game_data["home_team_id"],
            statbroadcast_gid=game_data["home_team_id"],
            team_name=game_data.get("home_team_name", game_data["home_team_id"]),
        )
    
    if game_data.get("away_team_id"):
        update_team_from_game(
            team_id=game_data["away_team_id"],
            statbroadcast_gid=game_data["away_team_id"],
            team_name=game_data.get("away_team_name", game_data["away_team_id"]),
        )
    
    return game_data


# ==================== Utility Functions ====================


def get_game_xml_path(game_id: str) -> Path:
    """Get the path to cached XML for a game."""
    return CACHE_DIR / f"game_{game_id}.xml"


def clear_xml_cache() -> int:
    """
    Clear all cached XML files.
    
    Returns:
        Number of files deleted
    """
    if not CACHE_DIR.exists():
        return 0
    
    count = 0
    for file in CACHE_DIR.glob("game_*.xml"):
        file.unlink()
        count += 1
    
    return count


def get_cached_game_ids() -> list[str]:
    """
    Get list of game IDs that have cached XML.
    
    Returns:
        List of game IDs
    """
    if not CACHE_DIR.exists():
        return []
    
    game_ids = []
    for file in CACHE_DIR.glob("game_*.xml"):
        # Extract game_id from filename like "game_16960.xml"
        match = re.search(r"game_(\d+)\.xml", file.name)
        if match:
            game_ids.append(match.group(1))
    
    return sorted(game_ids)


# ==================== Batch Processing ====================


def sync_games_for_date_range(
    start_date: str,
    end_date: str,
    max_games: int = 100
) -> dict:
    """
    Synchronize games for a date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD
        end_date: End date in YYYY-MM-DD
        max_games: Maximum games to process
    
    Returns:
        Summary dictionary
    """
    # Initialize database
    init_database()
    
    # Fetch game IDs
    games = fetch_game_ids_from_api(start_date, end_date)
    
    summary = {
        "total_games_found": len(games),
        "new_games_stored": 0,
        "already_existed": 0,
        "errors": 0,
    }
    
    for game in games[:max_games]:
        game_id = game["game_id"]
        
        if is_game_id_stored(game_id):
            summary["already_existed"] += 1
            continue
        
        try:
            # Try to fetch XML
            xml_content = fetch_game_xml(game_id, use_cache=True)
            
            if xml_content:
                summary["new_games_stored"] += 1
            else:
                summary["errors"] += 1
            
            # Rate limiting
            time.sleep(0.1)
        
        except Exception as e:
            print(f"Error processing game {game_id}: {e}")
            summary["errors"] += 1
    
    return summary


if __name__ == "__main__":
    # Initialize and show stats
    init_database()
    
    # Process sample game
    sample_data = process_sample_game()
    print(f"Processed sample game: {sample_data['game_id']}")
    print(f"  Home: {sample_data['home_team_name']} ({sample_data['home_team_id']})")
    print(f"  Away: {sample_data['away_team_name']} ({sample_data['away_team_id']})")
    
    # Show all stored game IDs
    all_games = get_all_stored_game_ids()
    print(f"\nStored games: {len(all_games)}")
    for game in all_games[:5]:
        print(f"  {game['game_id']}: {game['home_team_id']} vs {game['away_team_id']} on {game['game_date']}")
