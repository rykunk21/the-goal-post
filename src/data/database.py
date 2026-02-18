"""
Database operations for NCAAB prediction system.
Matches Steve.js schema from migrations 003 and 006.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Database path - located at ~/repos/ncaab-predictor/data/ncaab.db
DB_PATH = Path(__file__).parent.parent.parent / "data" / "ncaab.db"


def _get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a dictionary."""
    if row is None:
        return None
    return dict(row)


# ==================== Initialization ====================


def init_game_ids_table() -> None:
    """
    Create game_ids table if not exists.
    Schema per user requirements:
    - game_id: StatBroadcast game ID (PRIMARY KEY)
    - transition_probabilities_8: 8-dim vector as JSON
    - label_computed: BOOLEAN
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS game_ids (
            game_id TEXT PRIMARY KEY,
            sport TEXT NOT NULL DEFAULT 'mens-college-basketball',
            home_team_id TEXT,
            away_team_id TEXT,
            game_date DATE NOT NULL,
            processed BOOLEAN NOT NULL DEFAULT 0,
            
            -- InfoNCE training labels: 8-dim transition probability vector as JSON
            transition_probabilities_8 TEXT,
            
            -- Whether labels have been computed
            label_computed BOOLEAN NOT NULL DEFAULT 0,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
            FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_ids_home_team ON game_ids(home_team_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_ids_away_team ON game_ids(away_team_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_ids_processed ON game_ids(processed)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_ids_date ON game_ids(game_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_ids_sport ON game_ids(sport)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_ids_label_computed ON game_ids(label_computed)")
    
    conn.commit()
    conn.close()


def init_teams_table() -> None:
    """
    Create teams table if not exists.
    Schema per user requirements:
    - team_id: team identifier
    - statbroadcast_gid: StatBroadcast ID (PRIMARY KEY)
    - latent_vector_16: 16-dim latent vector as JSON array
    - last_updated: timestamp
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            statbroadcast_gid TEXT PRIMARY KEY,
            team_id TEXT NOT NULL,
            team_name TEXT NOT NULL,
            sport TEXT NOT NULL DEFAULT 'mens-college-basketball',
            conference TEXT,
            
            -- VAE latent representation: 16-dim vector as JSON array
            latent_vector_16 TEXT,
            
            -- Timestamps
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_teams_team_id ON teams(team_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_teams_sport ON teams(sport)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_teams_conference ON teams(conference)")
    
    conn.commit()
    conn.close()


def init_database() -> None:
    """Initialize all tables."""
    init_teams_table()
    init_game_ids_table()


# ==================== Game ID Operations ====================


def store_game_id_locally(
    game_id: str,
    home_team_id: Optional[str] = None,
    away_team_id: Optional[str] = None,
    game_date: Optional[str] = None,
    sport: str = "mens-college-basketball"
) -> bool:
    """
    Store a game ID in the local database.
    
    Args:
        game_id: The StatBroadcast/ESPN game ID
        home_team_id: Home team ID (e.g., "KEN" for Kentucky)
        away_team_id: Away team ID (e.g., "MSU" for Michigan State)
        game_date: Game date in YYYY-MM-DD format
        sport: Sport identifier
    
    Returns:
        True if stored successfully, False if already exists
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO game_ids (game_id, sport, home_team_id, away_team_id, game_date, processed, label_computed)
            VALUES (?, ?, ?, ?, ?, 0, 0)
        """, (game_id, sport, home_team_id, away_team_id, game_date))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Already exists
        return False
    finally:
        conn.close()


def get_all_stored_game_ids() -> list[dict]:
    """
    Retrieve all stored game IDs.
    
    Returns:
        List of game ID records as dictionaries
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT game_id, sport, home_team_id, away_team_id, game_date, 
               processed, label_computed, created_at, updated_at
        FROM game_ids
        ORDER BY game_date DESC, game_id
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    return [_row_to_dict(row) for row in rows]


def get_unprocessed_game_ids() -> list[str]:
    """
    Get list of game IDs that haven't been processed.
    
    Returns:
        List of unprocessed game IDs
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT game_id FROM game_ids WHERE processed = 0 ORDER BY game_date")
    rows = cursor.fetchall()
    conn.close()
    
    return [row[0] for row in rows]


def is_game_id_stored(game_id: str) -> bool:
    """
    Check if a game ID exists in the local database.
    
    Args:
        game_id: The game ID to check
    
    Returns:
        True if game exists, False otherwise
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT 1 FROM game_ids WHERE game_id = ?", (game_id,))
    result = cursor.fetchone()
    conn.close()
    
    return result is not None


def mark_game_processed(game_id: str) -> None:
    """Mark a game as processed."""
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE game_ids 
        SET processed = 1, updated_at = CURRENT_TIMESTAMP
        WHERE game_id = ?
    """, (game_id,))
    conn.commit()
    conn.close()


# ==================== Team Latent Operations ====================


def store_team_latent(
    team_id: str,
    latent_vector: list[float],
    statbroadcast_gid: Optional[str] = None,
    team_name: Optional[str] = None,
    sport: str = "mens-college-basketball"
) -> None:
    """
    Store a team's 16-dimensional latent representation.
    Uses statbroadcast_gid as primary key.
    
    Args:
        team_id: Team identifier (e.g., "KEN", "MSU")
        latent_vector: 16-dimensional latent vector
        statbroadcast_gid: StatBroadcast team ID (primary key)
        team_name: Full team name
        sport: Sport identifier
    """
    if len(latent_vector) != 16:
        raise ValueError(f"Latent vector must be 16 dimensions, got {len(latent_vector)}")
    
    # Use statbroadcast_gid as primary key
    gid = statbroadcast_gid or team_id
    
    conn = _get_connection()
    cursor = conn.cursor()
    
    # Store as simple JSON array
    latent_json = json.dumps(latent_vector)
    now = datetime.now().isoformat()
    
    cursor.execute("""
        INSERT INTO teams (statbroadcast_gid, team_id, team_name, sport, latent_vector_16, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(statbroadcast_gid) DO UPDATE SET
            team_id = excluded.team_id,
            team_name = excluded.team_name,
            sport = excluded.sport,
            latent_vector_16 = excluded.latent_vector_16,
            last_updated = excluded.last_updated
    """, (gid, team_id, team_name or team_id, sport, latent_json, now))
    
    conn.commit()
    conn.close()


def fetch_team_latent(statbroadcast_gid: str) -> Optional[list[float]]:
    """
    Retrieve a team's latent representation.
    
    Args:
        statbroadcast_gid: StatBroadcast team ID (primary key)
    
    Returns:
        16-dimensional latent vector or None if not found
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT latent_vector_16 
        FROM teams 
        WHERE statbroadcast_gid = ?
    """, (statbroadcast_gid,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row is None or row[0] is None:
        return None
    
    return json.loads(row[0])


def update_team_from_game(
    team_id: str,
    statbroadcast_gid: str,
    team_name: str,
    conference: Optional[str] = None
) -> None:
    """
    Update team metadata from game XML data.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO teams (team_id, statbroadcast_gid, team_name, conference, sport, last_synced)
        VALUES (?, ?, ?, ?, 'mens-college-basketball', CURRENT_TIMESTAMP)
        ON CONFLICT(team_id) DO UPDATE SET
            statbroadcast_gid = excluded.statbroadcast_gid,
            team_name = excluded.team_name,
            conference = COALESCE(excluded.conference, teams.conference),
            last_synced = CURRENT_TIMESTAMP
    """, (team_id, statbroadcast_gid, team_name, conference))
    
    conn.commit()
    conn.close()


# ==================== Transition Probabilities ====================


def store_transition_probs(
    game_id: str,
    home_probs: list[float],
    away_probs: list[float]
) -> None:
    """
    Store 8-dimensional transition probability vectors for a game.
    Stores both home and away as single JSON object.
    
    Args:
        game_id: The game ID
        home_probs: 8-dimensional transition probs for home team
        away_probs: 8-dimensional transition probs for away team
    """
    if len(home_probs) != 8 or len(away_probs) != 8:
        raise ValueError("Transition probabilities must be 8 dimensions")
    
    conn = _get_connection()
    cursor = conn.cursor()
    
    # Store as single JSON object containing both home and away
    probs_json = json.dumps({"home": home_probs, "away": away_probs})
    
    cursor.execute("""
        UPDATE game_ids 
        SET transition_probabilities_8 = ?,
            label_computed = 1,
            updated_at = CURRENT_TIMESTAMP
        WHERE game_id = ?
    """, (probs_json, game_id))
    
    conn.commit()
    conn.close()


def fetch_transition_probs(game_id: str) -> Optional[tuple[list[float], list[float]]]:
    """
    Retrieve transition probabilities for a game.
    
    Args:
        game_id: The game ID
    
    Returns:
        Tuple of (home_probs, away_probs) or None if not found
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT transition_probabilities_8
        FROM game_ids
        WHERE game_id = ?
    """, (game_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row is None or row[0] is None:
        return None
    
    probs = json.loads(row[0])
    return (probs.get("home", []), probs.get("away", []))


def get_games_with_labels() -> list[dict]:
    """Get all games that have transition probability labels."""
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT game_id, home_team_id, away_team_id, game_date
        FROM game_ids
        WHERE label_computed = 1
        ORDER BY game_date DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    return [_row_to_dict(row) for row in rows]


# ==================== Utility Functions ====================


def get_team_info(team_id: str) -> Optional[dict]:
    """Get full team information."""
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM teams WHERE team_id = ?
    """, (team_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    return _row_to_dict(row)


def get_all_teams() -> list[dict]:
    """Get all teams."""
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM teams ORDER BY team_name")
    rows = cursor.fetchall()
    conn.close()
    
    return [_row_to_dict(row) for row in rows]


def delete_game(game_id: str) -> bool:
    """Delete a game from the database."""
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM game_ids WHERE game_id = ?", (game_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return deleted


def get_database_stats() -> dict:
    """Get database statistics."""
    conn = _get_connection()
    cursor = conn.cursor()
    
    stats = {}
    
    # Count teams
    cursor.execute("SELECT COUNT(*) FROM teams")
    stats["teams_count"] = cursor.fetchone()[0]
    
    # Count games
    cursor.execute("SELECT COUNT(*) FROM game_ids")
    stats["games_count"] = cursor.fetchone()[0]
    
    # Count processed games
    cursor.execute("SELECT COUNT(*) FROM game_ids WHERE processed = 1")
    stats["processed_games"] = cursor.fetchone()[0]
    
    # Count games with labels
    cursor.execute("SELECT COUNT(*) FROM game_ids WHERE label_computed = 1")
    stats["games_with_labels"] = cursor.fetchone()[0]
    
    conn.close()
    return stats


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
    print(f"Database initialized at: {DB_PATH}")
    print(f"Stats: {get_database_stats()}")
