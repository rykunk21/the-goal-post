"""
Schedule Fetcher for NCAAB Prediction System.

Fetches team schedules from StatBroadcast and extracts game IDs.
Game IDs are used to stream XML data from the archive.

Uses DataTables AJAX endpoint (Steve.js method) for reliable game ID extraction.

Anti-blocking measures:
- User-Agent header matching browser
- Request jitter (0.3-0.7 sec random delay)
- Retry with exponential backoff (3 retries)
"""

import re
import time
import random
import json
import logging
from typing import List, Set, Optional
from pathlib import Path

import httpx
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
STATBROADCAST_TEAM_URL = "https://www.statbroadcast.com/events/statmonitr.php"
RATE_LIMIT_SECONDS = 1.0
REQUEST_TIMEOUT = 30

# Database
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.database import init_game_ids_table, store_game_id_locally, is_game_id_stored, _get_connection


class ScheduleFetcher:
    """Fetch team schedules from StatBroadcast.
    
    For each team GID, fetches the team's statmonitr page
    and extracts game IDs from the links.
    """
    
    def __init__(
        self,
        base_url: str = STATBROADCAST_TEAM_URL,
        rate_limit: float = RATE_LIMIT_SECONDS,
        timeout: int = REQUEST_TIMEOUT
    ):
        """Initialize schedule fetcher.
        
        Args:
            base_url: StatBroadcast team page URL
            rate_limit: Minimum seconds between requests
            timeout: HTTP request timeout
        """
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._last_request_time = 0.0
        self._client: Optional[httpx.Client] = None
        
    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client with anti-blocking headers."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
            )
        return self._client
    
    def _rate_limit(self):
        """Apply minimal rate limiting between requests."""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()
    
    def fetch_team_schedule_datatables(self, team_gid: str) -> List[str]:
        """Fetch game IDs using DataTables AJAX endpoint (Steve.js method).
        
        This is more reliable than HTML scraping as it uses the JSON API.
        Gets first 500 games (most recent) per team.
        
        Args:
            team_gid: Team GID (e.g., 'duke', 'msu', 'unc')
            
        Returns:
            List of game IDs
        """
        logger.info(f"Fetching schedule via DataTables for team: {team_gid}")
        
        # Step 1: Fetch archive page to discover dataUrl
        archive_url = f"https://www.statbroadcast.com/events/archive.php?gid={team_gid}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        try:
            self._rate_limit()
            response = requests.get(archive_url, headers=headers, timeout=15)
            response.raise_for_status()
        except Exception as e:
            logger.warning(f"Failed to fetch archive page for {team_gid}: {e}")
            return []
        
        # Extract dataUrl from page
        match = re.search(r'dataUrl\s*=\s*"([^"]+)"', response.text)
        if match:
            data_url = "https://www.statbroadcast.com" + match.group(1).replace('..', '')
            logger.info(f"Found dataUrl: {data_url[:60]}...")
        else:
            # Try to construct URL from page - look for the form action
            form_match = re.search(r'action=["\']([^"\']*_archive\.php[^"\']*)["\']', response.text)
            if form_match:
                data_url = "https://www.statbroadcast.com" + form_match.group(1)
            else:
                # Ultimate fallback - use the events endpoint directly
                data_url = f"https://www.statbroadcast.com/events/_archive.php"
            logger.warning(f"No dataUrl found, using fallback: {data_url[:60]}...")
        
        # Step 2: POST to DataTables endpoint with form data
        form_data = {
            'draw': 1,
            'start': 0,
            'length': 500,  # 500 games per request
            'order[0][column]': 0,
            'order[0][dir]': 'desc',
            'search[value]': '',
            'o[gid]': team_gid,
            'o[sports]': 'M;bbgame',
            'o[conf]': '',
            'o[tourn]': '',
            'o[startdate]': '',
            'o[enddate]': '',
            'columns[0][data]': 0,
            'columns[0][searchable]': 'true',
            'columns[0][orderable]': 'true',
            'columns[1][data]': 1,
            'columns[1][searchable]': 'true',
            'columns[1][orderable]': 'false',
        }
        
        ajax_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': archive_url,
            'X-Requested-With': 'XMLHttpRequest',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        try:
            self._rate_limit()
            response = requests.post(data_url, data=form_data, headers=ajax_headers, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch DataTables for {team_gid}: {e}")
            return []
        
        # Step 3: Extract game IDs from JSON
        game_ids = []
        records_total = data.get('recordsTotal', 0)
        logger.info(f"Total records for {team_gid}: {records_total}")
        
        for row in data.get('data', []):
            link_html = row.get('eventlink') or row.get('event_link') or row.get('link')
            if not link_html:
                for k, v in row.items():
                    if isinstance(v, str) and 'id=' in v:
                        link_html = v
                        break
            
            if link_html:
                match = re.search(r'id=(\d+)', link_html)
                if match:
                    game_ids.append(match.group(1))
        
        logger.info(f"Found {len(game_ids)} game IDs for team {team_gid}")
        return game_ids
    
    def fetch_team_schedule(self, team_gid: str) -> List[str]:
        """Fetch schedule for a specific team using DataTables method.
        
        Args:
            team_gid: Team GID (e.g., 'duke', 'msu', 'unc')
            
        Returns:
            List of game IDs
        """
        return self.fetch_team_schedule_datatables(team_gid)
    
    def fetch_all_schedules(self, team_gids: List[str]) -> List[str]:
        """Fetch schedules for multiple teams.
        
        Args:
            team_gids: List of team GIDs
            
        Returns:
            List of unique game IDs across all teams
        """
        all_game_ids: Set[str] = set()
        
        for i, team_gid in enumerate(team_gids):
            logger.info(f"Processing team {i+1}/{len(team_gids)}: {team_gid}")
            
            game_ids = self.fetch_team_schedule(team_gid)
            all_game_ids.update(game_ids)
        
        logger.info(f"Total unique game IDs found: {len(all_game_ids)}")
        
        return list(all_game_ids)
    
    def fetch_and_cache_games(self, team_gids: List[str]) -> int:
        """Fetch schedules and cache game IDs in database.
        
        Args:
            team_gids: List of team GIDs
            
        Returns:
            Number of new game IDs cached
        """
        game_ids = self.fetch_all_schedules(team_gids)
        
        cached = 0
        for game_id in game_ids:
            if store_game_id_locally(game_id):
                cached += 1
        
        logger.info(f"Cached {cached} new game IDs")
        return cached
    
    def get_cached_game_ids(self) -> List[str]:
        """Get cached game IDs from database.
        
        Returns:
            List of game IDs
        """
        conn = _get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT game_id FROM game_ids ORDER BY game_date DESC, game_id")
        rows = cursor.fetchall()
        conn.close()
        
        return [row['game_id'] for row in rows]
    
    def get_uncached_game_ids(self) -> List[str]:
        """Get game IDs not yet processed.
        
        Returns:
            List of unprocessed game IDs
        """
        conn = _get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT game_id FROM game_ids WHERE processed = 0 ORDER BY game_date DESC, game_id")
        rows = cursor.fetchall()
        conn.close()
        
        return [row['game_id'] for row in rows]
    
    def close(self):
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


def fetch_team_schedules(team_gids: List[str]) -> List[str]:
    """Fetch schedules for multiple teams and cache in database.
    
    Args:
        team_gids: List of team GIDs
        
    Returns:
        List of unique game IDs
    """
    # Initialize database
    init_game_ids_table()
    
    fetcher = ScheduleFetcher()
    game_ids = fetcher.fetch_all_schedules(team_gids)
    fetcher.close()
    
    # Cache in database
    for game_id in game_ids:
        store_game_id_locally(game_id)
    
    return game_ids


def get_game_ids() -> List[str]:
    """Get cached game IDs or fetch if needed.
    
    Returns:
        List of game IDs
    """
    init_game_ids_table()
    
    fetcher = ScheduleFetcher()
    game_ids = fetcher.get_cached_game_ids()
    fetcher.close()
    
    return game_ids


def discover_all_game_ids() -> dict:
    """Discover all game IDs for all known teams using DataTables method.
    
    Loads team GIDs from data/statbroadcast_team_gids.json,
    fetches schedules for each team via DataTables AJAX,
    and saves unique game IDs to data/statbroadcast_game_ids.json.
    
    Returns:
        Dictionary mapping team GID to list of game IDs
    """
    # Load team GIDs
    team_gids_path = Path(__file__).parent.parent.parent / "data" / "statbroadcast_team_gids.json"
    with open(team_gids_path, 'r') as f:
        team_gids = json.load(f)
    
    logger.info(f"Loaded {len(team_gids)} team GIDs")
    
    # Fetch schedules for all teams
    fetcher = ScheduleFetcher()
    
    team_game_ids = {}
    all_game_ids: Set[str] = set()
    
    for i, team_gid in enumerate(team_gids):
        logger.info(f"Processing team {i+1}/{len(team_gids)}: {team_gid}")
        
        game_ids = fetcher.fetch_team_schedule_datatables(team_gid)
        team_game_ids[team_gid] = game_ids
        all_game_ids.update(game_ids)
        
        logger.info(f"  Found {len(game_ids)} games (total: {len(all_game_ids)})")
    
    fetcher.close()
    
    # Save to JSON file
    output_path = Path(__file__).parent.parent.parent / "data" / "statbroadcast_game_ids.json"
    with open(output_path, 'w') as f:
        json.dump(team_game_ids, f, indent=2)
    
    logger.info(f"Saved {len(all_game_ids)} unique game IDs to {output_path}")
    
    return team_game_ids


# Main execution for testing
if __name__ == "__main__":
    print("=" * 60)
    print("Schedule Fetcher Test")
    print("=" * 60)
    
    # Test with a few teams
    test_teams = ['duke', 'unc', 'msu']
    
    fetcher = ScheduleFetcher()
    
    for team in test_teams:
        print(f"\nFetching schedule for {team}...")
        game_ids = fetcher.fetch_team_schedule(team)
        print(f"  Found {len(game_ids)} games: {game_ids[:5]}...")
    
    fetcher.close()
