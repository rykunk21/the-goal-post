#!/usr/bin/env python3
"""
Scrape StatBroadcast team GIDs from the events page.

This script fetches the StatBroadcast events page and extracts
all team GIDs (e.g., "duke", "msu") that can be used to fetch schedules.

Usage:
    python scrape_team_gids.py
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re


STATBROADCAST_EVENTS_URL = "https://www.statbroadcast.com/events/"


def scrape_team_gids():
    """Scrape team GIDs from StatBroadcast events page."""
    print(f"Fetching {STATBROADCAST_EVENTS_URL}...")
    
    response = requests.get(STATBROADCAST_EVENTS_URL, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links with gid parameter
    team_gids = set()
    
    for link in soup.find_all('a', href=True):
        href = link['href']
        
        # Look for patterns like ?gid=duke or ?gid=msu
        match = re.search(r'[?&]gid=([a-zA-Z0-9]+)', href)
        if match:
            gid = match.group(1).lower()
            # Filter out non-team GIDs
            skip_patterns = ['bcs', 'stat', 'admin', 'system', 'index', 'demo', 'support', 'blog', 'about']
            if gid not in skip_patterns and len(gid) >= 2:
                team_gids.add(gid)
    
    return sorted(team_gids)


def save_team_gids(team_gids, output_path="data/statbroadcast_team_gids.json"):
    """Save team GIDs to JSON file."""
    import os
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(team_gids, f, indent=2)
    
    print(f"Saved {len(team_gids)} team GIDs to {output_path}")


def main():
    """Main function."""
    print("="*60)
    print("StatBroadcast Team GID Scraper")
    print("="*60)
    
    team_gids = scrape_team_gids()
    
    print(f"\nFound {len(team_gids)} team GIDs:")
    print(f"  First 20: {team_gids[:20]}")
    if len(team_gids) > 20:
        print(f"  ... and {len(team_gids) - 20} more")
    
    # Save to file
    save_team_gids(team_gids)
    
    return team_gids


if __name__ == "__main__":
    main()
