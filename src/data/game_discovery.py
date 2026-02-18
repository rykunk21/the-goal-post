"""
Game Discovery for NCAAB Prediction System.

Generates and discovers valid game IDs from StatBroadcast archive.
Focuses on 2024-2025 NCAAB men's basketball season.
"""

import time
from typing import List, Optional, Set
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
RATE_LIMIT_SECONDS = 1.0
REQUEST_TIMEOUT = 10
STATBROADCAST_BASE = "http://archive.statbroadcast.com"

# Game ID ranges for NCAAB
# StatBroadcast uses alphanumeric team GIDs (e.g., "duke", "msu", "cal")
# The archive XML URLs use numeric game IDs (e.g., 29995)
# Known team GIDs (can be scraped from StatBroadcast events page)
KNOWN_TEAM_GIDS = [
    'duke', 'unc', 'ku', 'kentucky', 'villanova', 'michigan', 'msu',
    ' syracuse', 'arizona', 'ucla', 'texas', 'gonzaga', 'maryland', 'purdue',
    'northcarolina', 'florida', 'louisville', 'oregon', 'washington', 'arizona',
    'connecticut', 'cincinnati', 'michiganstate', 'ohiostate', 'wisconsin',
    'iowa', 'illinois', 'michigan', 'minnesota', 'nebraska', 'pennstate',
    'rutgers', 'indiana', 'northwestern', 'usc', 'stanford', 'california',
    'colorado', 'utah', 'washingtonstate', 'oregonstate', 'ucla', 'pepperdine',
    'saintmarys', 'sdsu', 'newmexico', 'nevada', 'boiseidaho', 'fresnostate',
    'hawaii', 'utahstate', 'byu', 'saintlouis', 'dayton', 'richmond', 'gw',
    'temple', 'xavier', 'creighton', 'marquette', 'georgetown', 'setonhall',
    'providence', 'villanova', 'stjohns', 'butler', 'depaul', 'loyolachi',
    'bradley', 'drake', 'wichita', 'northerniowa', 'indianastate', 'evansville',
    'southernillinois', 'illinoisstate', 'loyolai', 'missouristate', 'belmont',
    'moreheadstate', 'murraystate', 'tennesseestate', 'eastkentucky', 'morehead',
    'austinpeay', 'jacksontennessee', 'tennesseetech', 'chattanooga', 'furman',
    'wofford', 'uncgreensboro', 'charlotte', 'davidson', 'duquesne', 'la salle',
    'saintjosephs', 'lambeau', 'lehigh', 'navy', 'bucknell', 'army', 'bostoncollege',
    'georgiatech', 'virginia', 'virginiatech', 'northcarolinastate', 'clemson',
    'floridastate', 'miami', 'wakeforest', 'georgia', 'auburn', 'tennessee',
    'southcarolina', 'mississippistate', 'olemiss', 'arkansas', 'lsu', 'alabama',
    'oklahoma', 'oklahomastate', 'texastech', 'baylor', 'tcus', 'houston', 'memphis',
    'tulane', 'tulsa', 'smug', 'utsa', 'utep', 'rice'
]

# Game ID ranges (numeric - from archive)
# These are discovered by scraping team schedule pages
GAME_ID_RANGE_START = 29990
GAME_ID_RANGE_END = 32000

# Current season (2024-2025) - using latest available archive
CURRENT_SEASON_START = 29990
CURRENT_SEASON_END = 31000


class GameDiscovery:
    """Discover valid game IDs from StatBroadcast archive.
    
    Methods:
        - Scan ID range with HTTP HEAD requests
        - Generate likely valid IDs based on patterns
        - Parallel discovery for efficiency
    """
    
    def __init__(
        self,
        base_url: str = STATBROADCAST_BASE,
        rate_limit: float = RATE_LIMIT_SECONDS,
        timeout: int = REQUEST_TIMEOUT
    ):
        """Initialize game discovery.
        
        Args:
            base_url: StatBroadcast base URL
            rate_limit: Minimum seconds between requests
            timeout: HTTP request timeout
        """
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._last_request_time = 0.0
        
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()
    
    def check_game_exists(self, game_id: int) -> bool:
        """Check if a game ID exists using HTTP HEAD request.
        
        Args:
            game_id: Game ID to check
            
        Returns:
            True if game exists (200), False otherwise
        """
        url = f"{self.base_url}/{game_id}.xml"
        
        try:
            self._rate_limit()
            response = requests.head(url, timeout=self.timeout)
            return response.status_code == 200
            
        except requests.exceptions.RequestException:
            return False
    
    def discover_games_in_range(
        self,
        start_id: int,
        end_id: int,
        max_games: Optional[int] = None,
        progress: bool = True
    ) -> List[int]:
        """Discover valid game IDs in a range.
        
        Args:
            start_id: Starting game ID
            end_id: Ending game ID
            max_games: Maximum number of games to find
            progress: Whether to show progress
            
        Returns:
            List of valid game IDs
        """
        valid_ids = []
        total_checked = 0
        total_in_range = end_id - start_id + 1
        
        for game_id in range(start_id, end_id + 1):
            if max_games and len(valid_ids) >= max_games:
                break
                
            if self.check_game_exists(game_id):
                valid_ids.append(game_id)
                logger.info(f"Found valid game ID: {game_id} ({len(valid_ids)} total)")
            
            total_checked += 1
            
            if progress and total_checked % 100 == 0:
                pct = 100 * total_checked / total_in_range
                logger.info(f"Progress: {pct:.1f}% ({total_checked}/{total_in_range})")
        
        return valid_ids
    
    def discover_games_parallel(
        self,
        start_id: int,
        end_id: int,
        max_games: Optional[int] = None,
        num_workers: int = 5,
        progress: bool = True
    ) -> List[int]:
        """Discover valid game IDs in parallel.
        
        Uses ThreadPoolExecutor for concurrent HTTP requests.
        
        Args:
            start_id: Starting game ID
            end_id: Ending game ID
            max_games: Maximum number of games to find
            num_workers: Number of parallel workers
            progress: Whether to show progress
            
        Returns:
            List of valid game IDs
        """
        game_ids = list(range(start_id, end_id + 1))
        valid_ids = []
        
        def check_id(gid: int) -> Optional[int]:
            if self.check_game_exists(gid):
                return gid
            return None
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(check_id, gid): gid for gid in game_ids}
            
            completed = 0
            for future in as_completed(futures):
                if max_games and len(valid_ids) >= max_games:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
                
                result = future.result()
                if result is not None:
                    valid_ids.append(result)
                    logger.info(f"Found valid game ID: {result}")
                
                completed += 1
                
                if progress and completed % 100 == 0:
                    pct = 100 * completed / len(game_ids)
                    logger.info(f"Progress: {pct:.1f}% ({completed}/{len(game_ids)})")
        
        return sorted(valid_ids)
    
    def generate_season_game_ids(
        self,
        season: str = "2024-2025",
        max_games: int = 2000
    ) -> List[int]:
        """Generate likely game IDs for a season.
        
        Uses empirical patterns - typically:
        - Season starts around ID 15000-17000
        - ~5000-8000 games per season for D1
        
        Args:
            season: Season string (e.g., "2024-2025")
            max_games: Maximum games to generate
            
        Returns:
            List of likely valid game IDs
        """
        # Generate sequential IDs in likely range
        if season == "2024-2025":
            start = 15000
            end = 28000
        elif season == "2023-2024":
            start = 14000
            end = 25000
        else:
            # Default range
            start = CURRENT_SEASON_START
            end = CURRENT_SEASON_END
        
        # Generate list of IDs to check
        game_ids = list(range(start, end + 1))
        
        # Shuffle for more efficient discovery
        import random
        random.seed(42)
        random.shuffle(game_ids)
        
        # Limit to max_games
        return game_ids[:max_games]


def generate_game_ids_for_streaming(
    num_games: int = 1000,
    strategy: str = "range",
    season: str = "2024-2025"
) -> List[int]:
    """Generate game IDs for streaming.
    
    Args:
        num_games: Number of games to generate
        strategy: Generation strategy ("range", "random", "sequential")
        season: Season for ID mapping
        
    Returns:
        List of game IDs
    """
    if strategy == "range":
        # Use a broad range
        return list(range(GAME_ID_RANGE_START, GAME_ID_RANGE_START + num_games))
    
    elif strategy == "random":
        import random
        random.seed(42)
        return random.sample(
            range(GAME_ID_RANGE_START, GAME_ID_RANGE_END),
            num_games
        )
    
    elif strategy == "sequential":
        # Sequential with gaps (more realistic)
        base = CURRENT_SEASON_START
        return [base + i * 3 for i in range(num_games)]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def quick_discovery(num_games: int = 100) -> List[int]:
    """Quickly discover some valid game IDs.
    
    Does a quick scan to find valid IDs without full discovery.
    
    Args:
        num_games: Number of games to try to find
        
    Returns:
        List of valid game IDs found
    """
    discovery = GameDiscovery()
    
    # Quick scan of likely range
    logger.info(f"Quick scanning for {num_games} valid games...")
    
    valid_ids = []
    
    # Scan in chunks
    for start in range(15000, 25000, 100):
        if len(valid_ids) >= num_games:
            break
            
        for gid in range(start, start + 100):
            if len(valid_ids) >= num_games:
                break
                
            if discovery.check_game_exists(gid):
                valid_ids.append(gid)
                logger.info(f"Found: {gid}")
    
    return valid_ids


# Main execution for testing
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Game Discovery Test")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # Check specific game ID
        game_id = int(sys.argv[1])
        discovery = GameDiscovery()
        exists = discovery.check_game_exists(game_id)
        print(f"Game {game_id} exists: {exists}")
    else:
        # Quick discovery
        print("\nRunning quick discovery (50 games)...")
        valid_ids = quick_discovery(50)
        print(f"\nFound {len(valid_ids)} valid game IDs:")
        print(valid_ids[:20], "..." if len(valid_ids) > 20 else "")
