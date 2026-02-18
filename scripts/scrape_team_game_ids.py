#!/usr/bin/env python3
"""
StatBroadcast Team Schedule Scraper.

Discovers game IDs by scraping team schedule pages from StatBroadcast.
This is the CORRECT way to get game IDs for the XML archive.

Usage:
    python scrape_team_game_ids.py [--team=duke] [--output=data/team_game_ids.json]
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os
from pathlib import Path
from urllib.parse import urljoin
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATBROADCAST_EVENTS_URL = "https://www.statbroadcast.com/events/"
STATBROADCAST_ARCHIVE_URL = "https://www.statbroadcast.com/events/archive.php"


def scrape_team_gids():
    """Scrape team GIDs from StatBroadcast events page."""
    logger.info(f"Fetching {STATBROADCAST_EVENTS_URL}...")
    
    response = requests.get(STATBROADCAST_EVENTS_URL, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    team_gids = set()
    
    for link in soup.find_all('a', href=True):
        href = link['href']
        match = re.search(r'[?&]gid=([a-zA-Z0-9]+)', href)
        if match:
            gid = match.group(1).lower()
            skip_patterns = ['bcs', 'stat', 'admin', 'system', 'index', 'demo', 'support', 'blog', 'about']
            if gid not in skip_patterns and len(gid) >= 2:
                team_gids.add(gid)
    
    return sorted(team_gids)


def discover_game_ids_for_team(team_gid, max_games=500):
    """Discover game IDs for a specific team by fetching their archive page.
    
    Args:
        team_gid: StatBroadcast team GID (e.g., 'duke', 'msu')
        max_games: Maximum number of games to retrieve
        
    Returns:
        List of game IDs as strings
    """
    url = f"{STATBROADCAST_ARCHIVE_URL}?gid={team_gid}"
    
    try:
        # Fetch the archive page to get the dataUrl with time/hash
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract the dataUrl pattern from the page
        match = re.search(r'dataUrl\s*=\s*"([^"]+)"', response.text)
        if not match:
            logger.warning(f"Could not find dataUrl for team {team_gid}")
            return []
        
        # Get the dataUrl path
        data_url_rel = match.group(1)
        
        # Convert relative URL to absolute
        if data_url_rel.startswith('..'):
            data_url_rel = data_url_rel.replace('..', '')
        
        base_url = "https://www.statbroadcast.com"
        data_url = base_url + data_url_rel
        
        # Extract time and hash from the dataUrl
        time_match = re.search(r'time=(\d+)', data_url)
        hash_match = re.search(r'hash=([^&]+)', data_url)
        
        if not time_match or not hash_match:
            logger.warning(f"Could not extract time/hash from dataUrl for {team_gid}")
            return []
        
        # Now fetch the actual game data using the DataTables endpoint
        game_ids = []
        
        # First request to get total records
        params = {
            'draw': 1,
            'start': 0,
            'length': min(100, max_games),
            'order[0][column]': 0,
            'order[0][dir]': 'desc',
            'search[value]': '',
            'search[regex]': 'false',
            'columns[0][data]': 0,
            'columns[0][searchable]': 'true',
            'columns[0][orderable]': 'true',
            'columns[1][data]': 1,
            'columns[1][searchable]': 'true',
            'columns[2][data]': 2,
            'columns[2][searchable]': 'true',
            'o[gid]': team_gid,
            'o[conf]': '',
            'o[tourn]': '',
            'o[sports]': 'M;bbgame',  # Men's Basketball
            'o[startdate]': '',
            'o[enddate]': '',
            'o[members]': '',
            'o[champonly]': 0,
            'time': time_match.group(1),
            'hash': hash_match.group(1)
        }
        
        # Build the full URL
        full_url = f"{base_url}/events/_archive.php"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': url,
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'X-Requested-With': 'XMLHttpRequest'
        }
        
        # Make the POST request
        response = requests.post(full_url, data=params, headers=headers, timeout=30)
        
        if response.status_code != 200:
            logger.warning(f"Failed to fetch game data for {team_gid}: HTTP {response.status_code}")
            return []
        
        data = response.json()
        
        if not data or 'data' not in data:
            logger.warning(f"No data returned for {team_gid}")
            return []
        
        # Extract game IDs from the response
        for row in data.get('data', []):
            if isinstance(row, list) and len(row) >= 4:
                # Game ID is typically in the link
                link_cell = str(row[3]) if len(row) > 3 else str(row[2]) if len(row) > 2 else ''
                id_match = re.search(r'id=(\d+)', link_cell)
                if id_match:
                    game_ids.append(id_match.group(1))
            elif isinstance(row, dict):
                # Try to find ID in any field
                for key, value in row.items():
                    if isinstance(value, str):
                        id_match = re.search(r'id=(\d+)', value)
                        if id_match:
                            game_ids.append(id_match.group(1))
        
        logger.info(f"Found {len(game_ids)} game IDs for team {team_gid}")
        return game_ids
        
    except Exception as e:
        logger.error(f"Error discovering game IDs for {team_gid}: {e}")
        return []


def discover_all_game_ids(teams=None, max_per_team=200, rate_limit=1.0):
    """Discover game IDs for all teams.
    
    Args:
        teams: List of team GIDs (if None, scrapes from events page)
        max_per_team: Maximum games per team
        rate_limit: Seconds between requests
        
    Returns:
        Dict mapping team GID to list of game IDs
    """
    if teams is None:
        teams = scrape_team_gids()
    
    logger.info(f"Discovering game IDs for {len(teams)} teams...")
    
    all_game_ids = {}
    
    for i, team_gid in enumerate(teams):
        logger.info(f"[{i+1}/{len(teams)}] Processing team: {team_gid}")
        
        game_ids = discover_game_ids_for_team(team_gid, max_games=max_per_team)
        
        if game_ids:
            all_game_ids[team_gid] = game_ids
        
        # Rate limiting
        if i < len(teams) - 1:
            time.sleep(rate_limit)
    
    # Flatten to get unique game IDs
    unique_game_ids = set()
    for gid_list in all_game_ids.values():
        unique_game_ids.update(gid_list)
    
    logger.info(f"Total unique game IDs discovered: {len(unique_game_ids)}")
    
    return all_game_ids, sorted(unique_game_ids)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape StatBroadcast game IDs')
    parser.add_argument('--team', type=str, default=None, help='Specific team GID to scrape')
    parser.add_argument('--output', type=str, default='data/statbroadcast_game_ids.json', help='Output file')
    parser.add_argument('--teams-file', type=str, default='data/statbroadcast_team_gids.json', help='Teams file')
    parser.add_argument('--max-per-team', type=int, default=200, help='Max games per team')
    args = parser.parse_args()
    
    print("="*60)
    print("StatBroadcast Game ID Discovery")
    print("="*60)
    
    if args.team:
        # Scrape specific team
        game_ids = discover_game_ids_for_team(args.team, max_games=args.max_per_team)
        print(f"\nFound {len(game_ids)} game IDs for {args.team}")
        print(f"  Sample: {game_ids[:10]}...")
        
        # Save
        output = {args.team: game_ids}
    else:
        # Load team GIDs from file or scrape
        teams_file = Path(args.teams_file)
        if teams_file.exists():
            with open(teams_file) as f:
                teams = json.load(f)
            print(f"Loaded {len(teams)} teams from {teams_file}")
        else:
            teams = scrape_team_gids()
        
        # Discover game IDs
        team_games, unique_games = discover_all_game_ids(teams, max_per_team=args.max_per_team)
        
        print(f"\nTotal unique game IDs: {len(unique_games)}")
        print(f"  Sample: {unique_games[:20]}...")
        
        output = {
            'teams': team_games,
            'unique_game_ids': unique_games
        }
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
