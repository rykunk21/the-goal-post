# Steve.js Database Schema Documentation

## Overview

The NCAAB predictor system references Steve.js database schema to ensure compatibility with the InfoNCE architecture for basketball prediction. This document outlines the key tables and fields used.

## Source

Steve.js migrations are located at: `~/repos/steve.js/src/database/migrations/`

---

## Key Tables

### 1. `teams` Table

Located in: `003_create_teams_and_games.sql`

**Purpose**: Store team information and latent representations.

**Schema**:
```sql
CREATE TABLE teams (
    team_id TEXT PRIMARY KEY,           -- Team identifier (e.g., "KEN", "MSU")
    statbroadcast_gid TEXT NOT NULL,   -- StatBroadcast team ID
    team_name TEXT NOT NULL,           -- Full team name
    sport TEXT NOT NULL DEFAULT 'mens-college-basketball',
    conference TEXT,
    
    -- VAE posterior latent representation (JSON blob)
    -- Structure: {"mu": [16-dim], "sigma": [16-dim], "games_processed": int, "last_season": "2023-24"}
    statistical_representation TEXT,
    
    -- Player roster for injury analysis (JSON array)
    player_roster TEXT,
    
    -- Sync tracking
    last_synced TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Migration 006 additions
    model_version TEXT DEFAULT 'v1.0',
    representation_type TEXT DEFAULT 'bayesian_posterior',
    
    UNIQUE(statbroadcast_gid, sport)
);
```

**Indexes**:
- `idx_teams_statbroadcast_gid` - For looking up teams by StatBroadcast ID
- `idx_teams_sport` - Filter by sport
- `idx_teams_conference` - Filter by conference
- `idx_teams_last_synced` - Find teams needing sync
- `idx_teams_representation_type` - Filter by representation type
- `idx_teams_model_version` - Filter by model version

---

### 2. `game_ids` Table

Located in: `003_create_teams_and_games.sql`

**Purpose**: Store game IDs and InfoNCE training labels.

**Schema**:
```sql
CREATE TABLE game_ids (
    game_id TEXT PRIMARY KEY,            -- Game ID (from StatBroadcast/ESPN)
    sport TEXT NOT NULL DEFAULT 'mens-college-basketball',
    home_team_id TEXT,                  -- Reference to teams.team_id
    away_team_id TEXT,                  -- Reference to teams.team_id
    game_date DATE NOT NULL,
    processed BOOLEAN NOT NULL DEFAULT 0,
    
    -- InfoNCE training labels: transition probability vectors
    -- 8-dimensional vectors for contrastive learning
    transition_probabilities_home BLOB,
    transition_probabilities_away BLOB,
    labels_extracted BOOLEAN NOT NULL DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);
```

**Indexes**:
- `idx_game_ids_home_team` - Find games by home team
- `idx_game_ids_away_team` - Find games by away team
- `idx_game_ids_processed` - Find unprocessed games
- `idx_game_ids_date` - Games by date
- `idx_game_ids_sport` - Filter by sport
- `idx_game_ids_labels_extracted` - Games with/without labels

---

### 3. `vae_model_weights` Table

Located in: `003_create_teams_and_games.sql`

**Purpose**: Store frozen VAE model weights for InfoNCE latent space.

**Schema**:
```sql
CREATE TABLE vae_model_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    encoder_weights BLOB NOT NULL,
    decoder_weights BLOB,
    latent_dim INTEGER NOT NULL DEFAULT 16,
    input_dim INTEGER NOT NULL DEFAULT 80,
    training_completed BOOLEAN NOT NULL DEFAULT 0,
    frozen BOOLEAN NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Fields Stored Locally vs Fetched from API

### Stored Locally (in SQLite)

1. **Team Data**:
   - `team_id`, `statbroadcast_gid`, `team_name`
   - `conference`
   - `statistical_representation` (16-dim latent vector)
   - `last_synced` timestamp

2. **Game Data**:
   - `game_id`, `home_team_id`, `away_team_id`
   - `game_date`
   - `processed` flag
   - `transition_probabilities_home/away` (8-dim vectors)

### Fetched from XML API (StatBroadcast)

The full game XML contains detailed play-by-play data:

1. **Game Metadata**:
   - `gameid`, `sbid`, `competitionid`
   - `date`, `location`, `time`
   - `homeid`, `homename`, `visid`, `visname`

2. **Team Statistics**:
   - `linescore`: Period-by-period scores
   - `totals/stats`: FG%, 3P%, FT%, rebounds, assists, turnovers, etc.
   - `totals/special`: Points off turnovers, paint points, fast break points

3. **Player Statistics**:
   - Individual player stats per game
   - Per-period breakdowns
   - Season aggregates

4. **Play-by-Play**:
   - Detailed `plays` element with every event
   - Scoring plays, fouls, substitutions, rebounds

---

## Transition Probabilities

The InfoNCE architecture uses **8-dimensional transition probability vectors**:

- `transition_probabilities_home[0-3]`: Home team transition probs from states 0-3
- `transition_probabilities_home[4-7]`: Home team transition probs from states 4-7
- Same structure for away team

These are stored as BLOB/JSON in the database and extracted from the XML play-by-play data.

---

## Latent Representation

Teams are represented as **16-dimensional latent vectors** stored in `statistical_representation` as JSON:

```json
{
  "mu": [0.1, -0.2, ..., 0.05],   // 16 values
  "sigma": [1.0, 1.0, ..., 1.0], // 16 values
  "games_processed": 25,
  "last_season": "2024-25",
  "model_version": "v1.0",
  "type": "bayesian_posterior"
}
```

---

## XML Data Source

Sample XML structure from `sample_game_16960.xml`:

```xml
<bbgame source="StatBroadcast Listener" version="1.0">
  <venue gameid="16960" sbid="623619" competitionid="41097"
          homeid="KEN" homename="Kentucky"
          visid="MSU" visname="Michigan St."
          date="11/18/2025" location="Madison Square Garden - New York">
  </venue>
  <status complete="Y" period="2" periodtype="REGULAR" clock="00:00"></status>
  <team vh="H" id="KEN" name="Kentucky">
    <linescore line="27,39" score="66">
      <lineprd prd="1" score="27"></lineprd>
      <lineprd prd="2" score="39"></lineprd>
    </linescore>
    <totals>
      <stats fgm="20" fga="57" fgm3="7" fga3="30" ftm="19" fta="22" tp="66" ...></stats>
    </totals>
  </team>
  <team vh="V" id="MSU" name="Michigan St.">...</team>
</bbgame>
```

---

## Implementation Notes

1. **Database Location**: `~/repos/ncaab-predictor/data/ncaab.db`

2. **Key Functions**:
   - `init_database()`: Create all tables
   - `store_game_id_locally()`: Store game with metadata
   - `store_team_latent()`: Store 16-dim latent vector
   - `store_transition_probs()`: Store 8-dim transition probs
   - `fetch_game_xml()`: Fetch XML from API

3. **Data Flow**:
   1. Fetch game IDs from ESPN/StatBroadcast API
   2. Store game IDs locally
   3. Fetch XML for each game
   4. Parse team data → update `teams` table
   5. Extract transition probs → update `game_ids` table
   6. Generate/update team latent representations

---

*Last Updated: 2026-02-16*
