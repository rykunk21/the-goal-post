"""
Team GID Discovery for NCAAB Prediction System.

Scrapes team GIDs from StatBroadcast events page.
Team GIDs are used to fetch team schedules and discover game IDs.
"""

import re
import time
import logging
from typing import List, Dict, Optional, Set
from pathlib import Path

import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
STATBROADCAST_EVENTS_URL = "https://www.statbroadcast.com/events/all.php"
RATE_LIMIT_SECONDS = 1.0
REQUEST_TIMEOUT = 30

# Database
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.database import init_teams_table, _get_connection


class TeamGIDDiscovery:
    """Discover team GIDs from StatBroadcast.
    
    Scrapes the events page to extract team identifiers used
    for fetching team schedules and game archives.
    """
    
    def __init__(
        self,
        base_url: str = STATBROADCAST_EVENTS_URL,
        rate_limit: float = RATE_LIMIT_SECONDS,
        timeout: int = REQUEST_TIMEOUT
    ):
        """Initialize team GID discovery.
        
        Args:
            base_url: StatBroadcast events page URL
            rate_limit: Minimum seconds between requests
            timeout: HTTP request timeout
        """
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._last_request_time = 0.0
        self._client: Optional[httpx.Client] = None
        
    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()
    
    def _extract_team_gids(self, html: str) -> List[Dict[str, str]]:
        """Extract team GIDs from HTML.
        
        Looks for patterns like:
        - statmonitr.php?gid=duke
        - statmonitr.php?gid=msu
        - statmonitr.php?gid=ariz
        
        Args:
            html: HTML content
            
        Returns:
            List of dicts with 'gid' and 'name' keys
        """
        teams = []
        
        # Pattern 1: statmonitr.php?gid=TEAM
        pattern1 = r'<a[^>]+href=["\']statmonitr\.php\?gid=([a-zA-Z0-9]+)["\'][^>]*>([^<]+)</a>'
        matches1 = re.findall(pattern1, html)
        
        for gid, name in matches1:
            # Clean up the name
            name = name.strip()
            if name and gid:
                teams.append({
                    'gid': gid.lower(),
                    'name': name,
                    'source': 'statmonitr'
                })
        
        # Deduplicate by gid
        seen = set()
        unique_teams = []
        for team in teams:
            if team['gid'] not in seen:
                seen.add(team['gid'])
                unique_teams.append(team)
        
        return unique_teams
    
    def discover_team_gids(self) -> List[Dict[str, str]]:
        """Discover all team GIDs from StatBroadcast events page.
        
        Returns:
            List of dicts with 'gid' and 'name' keys
        """
        logger.info(f"Fetching team GIDs from {self.base_url}")
        
        client = self._get_client()
        self._rate_limit()
        
        response = client.get(self.base_url)
        response.raise_for_status()
        
        teams = self._extract_team_gids(response.text)
        
        logger.info(f"Discovered {len(teams)} team GIDs")
        
        return teams
    
    def save_to_database(self, teams: List[Dict[str, str]]) -> int:
        """Save discovered teams to database.
        
        Args:
            teams: List of team dicts with 'gid' and 'name'
            
        Returns:
            Number of teams saved
        """
        conn = _get_connection()
        cursor = conn.cursor()
        
        saved = 0
        for team in teams:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO teams (team_id, statbroadcast_gid, team_name, sport)
                    VALUES (?, ?, ?, 'mens-college-basketball')
                """, (team['gid'], team['gid'], team['name']))
                
                if cursor.rowcount > 0:
                    saved += 1
                    
            except Exception as e:
                logger.warning(f"Error saving team {team['gid']}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {saved} new teams to database")
        return saved
    
    def get_cached_teams(self) -> List[Dict[str, str]]:
        """Get teams from database cache.
        
        Returns:
            List of team dicts
        """
        conn = _get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT team_id, statbroadcast_gid, team_name
            FROM teams
            WHERE sport = 'mens-college-basketball'
            ORDER BY team_name
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'gid': row['statbroadcast_gid'],
                'name': row['team_name'],
                'team_id': row['team_id']
            }
            for row in rows
        ]
    
    def close(self):
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


def discover_and_cache_teams() -> List[Dict[str, str]]:
    """Discover teams and save to database.
    
    Returns:
        List of discovered teams
    """
    # Initialize database
    init_teams_table()
    
    # Discover teams
    discovery = TeamGIDDiscovery()
    teams = discovery.discover_team_gids()
    
    # Save to database
    discovery.save_to_database(teams)
    
    discovery.close()
    
    return teams


def get_teams() -> List[Dict[str, str]]:
    """Get cached teams or discover if needed.
    
    Returns:
        List of team dicts
    """
    init_teams_table()
    
    discovery = TeamGIDDiscovery()
    teams = discovery.get_cached_teams()
    
    if not teams:
        logger.info("No cached teams found, discovering...")
        teams = discover_and_cache_teams()
    
    discovery.close()
    
    return teams


# Main execution for testing
if __name__ == "__main__":
    print("=" * 60)
    print("Team GID Discovery Test")
    print("=" * 60)
    
    teams = discover_and_cache_teams()
    
    print(f"\nDiscovered {len(teams)} teams:")
    for team in teams[:20]:
        print(f"  {team['gid']}: {team['name']}")
    
    if len(teams) > 20:
        print(f"  ... and {len(teams) - 20} more")
