"""
Streaming XML Loader for NCAAB Prediction System.

Fetches game XML data from StatBroadcast archive and extracts 80-dim features.
Uses proper GID → Schedule → Game ID discovery instead of random iteration.

Rate limited to 1 request/second (StatBroadcast requirement).

Anti-blocking measures:
- User-Agent header matching browser
- Request jitter (1-2 sec random delay)
- Retry with exponential backoff (3 retries)
- HTTP (not HTTPS) for archive URLs
"""

import time
import random
import xml.etree.ElementTree as ET
from typing import Tuple, Optional, Dict, List, Set
import numpy as np
import logging
from pathlib import Path

import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting
RATE_LIMIT_SECONDS = 1.0
REQUEST_TIMEOUT = 30

# StatBroadcast base URL
STATBROADCAST_BASE = "http://archive.statbroadcast.com"

# Database imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.database import init_game_ids_table, store_game_id_locally, is_game_id_stored


class StreamingXMLoader:
    """Streaming XML loader for fetching game data from StatBroadcast.
    
    Features:
        - Rate limiting (1 req/sec)
        - GID-based game discovery (no random ID iteration)
        - In-memory XML parsing (no local storage)
        - 80-dim feature extraction
        - Graceful error handling
    """
    
    def __init__(
        self,
        base_url: str = STATBROADCAST_BASE,
        rate_limit: float = RATE_LIMIT_SECONDS,
        timeout: int = REQUEST_TIMEOUT,
        cache_dir: Optional[str] = None
    ):
        """Initialize streaming XML loader.
        
        Args:
            base_url: StatBroadcast base URL
            rate_limit: Minimum seconds between requests
            timeout: HTTP request timeout
            cache_dir: Optional directory for caching (not used currently)
        """
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.cache_dir = cache_dir
        self._last_request_time = 0.0
        self._client: Optional[httpx.Client] = None
        
    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client with anti-blocking headers."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
            )
        return self._client
    
    def _rate_limit(self):
        """Apply rate limiting between requests with jitter."""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self.rate_limit:
            sleep_time = (self.rate_limit - elapsed) + random.random()  # Add 0-1 sec jitter
            time.sleep(sleep_time)
        # Also add random delay before each request (1-2 sec)
        time.sleep(1.0 + random.random())  # 1-2 sec
        self._last_request_time = time.time()
    
    def _fetch_xml(self, game_id: int) -> Optional[str]:
        """Fetch XML content for a game ID with retry logic.
        
        Args:
            game_id: StatBroadcast game ID
            
        Returns:
            XML content as string, or None if not found
        """
        # Use HTTP (not HTTPS) to avoid blocks
        url = f"http://archive.statbroadcast.com/{game_id}.xml"
        
        # Retry with exponential backoff (3 retries)
        for attempt in range(3):
            try:
                client = self._get_client()
                self._rate_limit()
                response = client.get(url)
                
                if response.status_code == 404:
                    logger.debug(f"Game {game_id} not found (404)")
                    return None
                
                if response.status_code == 403:
                    logger.warning(f"Game {game_id} blocked (403 Forbidden)")
                    # Wait longer on 403
                    time.sleep(2 ** (attempt + 2))  # 4, 8, 16 sec
                    continue
                
                response.raise_for_status()
                return response.text
                
            except httpx.HTTPError as e:
                logger.warning(f"Error fetching game {game_id} (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)  # 1, 2, 4 sec
                continue
            except Exception as e:
                logger.warning(f"Unexpected error fetching game {game_id}: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                continue
        
        return None
    
    def _parse_xml(self, xml_content: str) -> Optional[ET.Element]:
        """Parse XML content into ElementTree.
        
        Args:
            xml_content: Raw XML string
            
        Returns:
            Root element or None if parsing fails
        """
        try:
            return ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.warning(f"XML parse error: {e}")
            return None
    
    def _extract_team_stats(self, root: ET.Element, team_id: str) -> Dict[str, float]:
        """Extract team statistics from XML root.
        
        Args:
            root: XML root element
            team_id: Team identifier
            
        Returns:
            Dictionary of team statistics
        """
        stats = {}
        
        # Find team element
        team_elem = None
        for team in root.findall('.//team'):
            if team.get('id') == team_id or team.get('id', '').upper() == team_id.upper():
                team_elem = team
                break
        
        if team_elem is None:
            return stats
        
        # Extract totals stats
        totals = team_elem.find('.//totals/stats')
        if totals is not None:
            stat_map = {
                'points': 'tp', 'fgm': 'fgm', 'fga': 'fga',
                'fg3m': 'fgm3', 'fg3a': 'fga3',
                'ftm': 'ftm', 'fta': 'fta',
                'oreb': 'oreb', 'dreb': 'dreb', 'rebounds': 'treb',
                'assists': 'ast', 'turnovers': 'to', 'steals': 'stl',
                'blocks': 'blk', 'fouls': 'pf'
            }
            
            for key, attr in stat_map.items():
                val = totals.get(attr)
                if val is not None:
                    try:
                        stats[key] = float(val)
                    except ValueError:
                        stats[key] = 0.0
        
        # Extract special stats
        special = team_elem.find('.//totals/special')
        if special is not None:
            special_map = {
                'possessions': 'poss_count',
                'paint_pts': 'pts_paint',
                'fastbreak_pts': 'pts_fastb',
                'bench_pts': 'pts_bench',
                'second_chance_pts': 'pts_ch2',
                'pts_off_to': 'pts_to'
            }
            
            for key, attr in special_map.items():
                val = special.get(attr)
                if val is not None:
                    try:
                        stats[key] = float(val)
                    except ValueError:
                        stats[key] = 0.0
        
        # Extract half scores if available
        for half in ['first', 'second', 'overtime']:
            half_elem = team_elem.find(f".//half[@number='{half}']")
            if half_elem is not None:
                pts = half_elem.get('points')
                if pts is not None:
                    try:
                        stats[f'{half}_pts'] = float(pts)
                    except ValueError:
                        pass
        
        return stats
    
    def _get_team_ids(self, root: ET.Element) -> Tuple[Optional[str], Optional[str]]:
        """Extract home and away team IDs from XML.
        
        Args:
            root: XML root element
            
        Returns:
            Tuple of (home_team_id, away_team_id)
        """
        teams = root.findall('.//team')
        
        if len(teams) >= 2:
            # Assume first is home, second is away (common convention)
            home_id = teams[0].get('id')
            away_id = teams[1].get('id')
            return home_id, away_id
        
        return None, None
    
    def fetch_game_features(self, game_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Fetch and extract features for a game.
        
        Args:
            game_id: StatBroadcast game ID
            
        Returns:
            Tuple of (home_features, away_features), each 80-dim numpy array
            Returns (None, None) if game not found or parse error
        """
        # Fetch XML
        xml_content = self._fetch_xml(game_id)
        if xml_content is None:
            return None, None
        
        # Parse XML
        root = self._parse_xml(xml_content)
        if root is None:
            return None, None
        
        # Get team IDs
        home_id, away_id = self._get_team_ids(root)
        
        # Extract stats for both teams
        home_stats = self._extract_team_stats(root, home_id) if home_id else {}
        away_stats = self._extract_team_stats(root, away_id) if away_id else {}
        
        # Build feature vectors
        home_features = self._build_feature_vector(home_stats, away_stats)
        away_features = self._build_feature_vector(away_stats, home_stats)
        
        # Log success
        logger.info(f"Successfully processed game {game_id}: {home_id} vs {away_id}")
        
        return home_features, away_features
    
    def _build_feature_vector(
        self,
        team_stats: Dict[str, float],
        opponent_stats: Dict[str, float]
    ) -> np.ndarray:
        """Build 80-dimensional feature vector from team stats.
        
        Uses the same feature extraction logic as feature_vector.py
        but operates on streaming data.
        
        Args:
            team_stats: Team's statistics
            opponent_stats: Opponent's statistics
            
        Returns:
            80-dimensional numpy array
        """
        from src.data.feature_vector import build_feature_vector
        
        return build_feature_vector(team_stats, opponent_stats)
    
    def fetch_batch(self, game_ids: List[int]) -> Tuple[list, list]:
        """Fetch features for multiple games.
        
        Args:
            game_ids: List of game IDs
            
        Returns:
            Tuple of (home_features_list, away_features_list)
        """
        home_features = []
        away_features = []
        
        for game_id in game_ids:
            home_feat, away_feat = self.fetch_game_features(game_id)
            if home_feat is not None and away_feat is not None:
                home_features.append(home_feat)
                away_features.append(away_feat)
        
        return home_features, away_features
    
    def close(self):
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


def discover_game_ids_from_teams(team_gids: List[str]) -> List[int]:
    """Discover game IDs from team schedules.
    
    Uses the proper flow: Team GID → Team Page → Game IDs
    
    Args:
        team_gids: List of team GIDs to fetch schedules for
        
    Returns:
        List of unique game IDs (as integers)
    """
    from src.data.schedule_fetcher import ScheduleFetcher
    
    logger.info(f"Discovering game IDs from {len(team_gids)} teams...")
    
    fetcher = ScheduleFetcher()
    game_ids_str = fetcher.fetch_all_schedules(team_gids)
    fetcher.close()
    
    # Convert to integers
    game_ids = [int(gid) for gid in game_ids_str]
    
    logger.info(f"Discovered {len(game_ids)} total game IDs")
    
    return game_ids


def get_game_ids_for_streaming(use_cache: bool = True) -> List[int]:
    """Get game IDs for streaming.
    
    Either loads from cache or discovers from team schedules.
    
    Args:
        use_cache: If True, use cached game IDs from database
        
    Returns:
        List of game IDs (as integers)
    """
    init_game_ids_table()
    
    # Check database for cached game IDs
    conn = __import__('sqlite3').connect(str(Path(__file__).parent.parent.parent / 'data' / 'ncaab.db'))
    conn.row_factory = __import__('sqlite3').Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT game_id FROM game_ids ORDER BY game_date DESC, game_id")
    rows = cursor.fetchall()
    conn.close()
    
    cached_ids = [int(row['game_id']) for row in rows]
    
    if cached_ids and use_cache:
        logger.info(f"Using {len(cached_ids)} cached game IDs")
        return cached_ids
    
    # Need to discover from teams
    logger.info("No cached game IDs, discovering from teams...")
    
    from src.data.team_gid_discovery import get_teams
    
    teams = get_teams()
    team_gids = [t['gid'] for t in teams]
    
    logger.info(f"Found {len(team_gids)} teams in database")
    
    # Get game IDs from team schedules
    game_ids = discover_game_ids_from_teams(team_gids)
    
    # Cache them
    for gid in game_ids:
        store_game_id_locally(str(gid))
    
    logger.info(f"Cached {len(game_ids)} game IDs in database")
    
    return game_ids


# Helper function for quick testing
def test_loader():
    """Test the streaming loader with a known game ID."""
    loader = StreamingXMLoader()
    
    # Try a few game IDs that we know exist
    test_ids = [611456, 614098, 649898, 649112]
    
    for game_id in test_ids:
        print(f"\nTesting game {game_id}...")
        home, away = loader.fetch_game_features(game_id)
        
        if home is not None:
            print(f"  Home features shape: {home.shape}")
            print(f"  Away features shape: {away.shape}")
            print(f"  Home features (first 10): {home[:10]}")
        else:
            print(f"  Game {game_id} not found or error")
    
    loader.close()


if __name__ == "__main__":
    test_loader()
