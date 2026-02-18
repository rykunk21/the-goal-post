# StatBroadcast XML Schema Documentation

## Overview

This document maps StatBroadcast XML game data fields to VAE input features for NCAAB prediction models.

## Source File

- **Primary sample:** `sample_game_16960.xml` (Michigan St. vs Kentucky, Nov 18, 2025)
- **Original location:** `/home/admin/repos/steve.js/tests/fixtures/statbroadcast-game-sample.xml`

## XML Structure

### Root Element: `<bbgame>`

| Attribute | Description | Example |
|-----------|-------------|---------|
| `source` | Data source | "StatBroadcast Listener" |
| `version` | Schema version | "1.0" |
| `generated` | Generation date | "11/18/2025" |
| `coords` | Coordinates flag | "N" |

---

## Section 1: `<venue>` - Game Information

**Path:** `/bbgame/venue`

| Attribute | Description | VAE Mapping |
|-----------|-------------|-------------|
| `gameid` | Unique game ID | Game identifier |
| `competitionid` | Tournament/league ID | Context feature |
| `date` | Game date | Season context |
| `visid` | Visitor team ID | Team ID (visitor) |
| `homeid` | Home team ID | Team ID (home) |
| `visname` | Visitor team name | Team name |
| `homename` | Home team name | Team name |
| `neutralgame` | Neutral site flag | Location feature |

---

## Section 2: `<status>` - Game Status

**Path:** `/bbgame/status`

| Attribute | Description | VAE Mapping |
|-----------|-------------|-------------|
| `complete` | Game completed (Y/N) | Data quality flag |
| `period` | Current period | Game progress |
| `periodtype` | REGULAR/OT | Game type |
| `clock` | Game clock | Time context |
| `gamestatus` | COMPLETE, LIVE, etc. | Status |

---

## Section 3: `<team>` - Team Statistics

**Path:** `/bbgame/team[@vh='V']` (visitor) and `/bbgame/team[@vh='H']` (home)

### 3.1 `<linescore>` - Score Summary

| Attribute | Description | VAE Mapping |
|-----------|-------------|-------------|
| `score` | Final score | **Target variable** |
| `line` | Score by half | Scoring pattern |
| `lineprd/@score` | Score per period | Tempo indicator |

### 3.2 `<totals>/<stats>` - Team Totals

This is the **primary source for VAE input features** (40 features per team):

| XML Attribute | Description | Units | VAE Feature Index |
|--------------|-------------|-------|-------------------|
| `fgm` | Field Goals Made | count | 0 |
| `fga` | Field Goals Attempted | count | 1 |
| `fgpct` | FG% | percent | 2 |
| `fgm3` | 3-Point Made | count | 3 |
| `fga3` | 3-Point Attempted | count | 4 |
| `fg3pct` | 3P% | percent | 5 |
| `ftm` | Free Throws Made | count | 6 |
| `fta` | Free Throws Attempted | count | 7 |
| `ftpct` | FT% | percent | 8 |
| `tp` | Total Points | count | 9 |
| `blk` | Blocks | count | 10 |
| `stl` | Steals | count | 11 |
| `ast` | Assists | count | 12 |
| `oreb` | Offensive Rebounds | count | 13 |
| `dreb` | Defensive Rebounds | count | 14 |
| `treb` | Total Rebounds | count | 15 |
| `pf` | Personal Fouls | count | 16 |
| `to` | Turnovers | count | 17 |
| `drawn` | Fouls Drawn | count | 18 |
| `deadball` | Deadball rebounds | count | 19 |

### 3.3 `<totals>/<special>` - Special Statistics

Extended team metrics (20+ features):

| XML Attribute | Description | VAE Mapping |
|--------------|-------------|-------------|
| `pts_to` | Points off turnovers | Efficiency |
| `pts_ch2` | Points in the paint (close) | Interior scoring |
| `pts_paint` | Total paint points | Interior game |
| `pts_fastb` | Fast break points | Transition offense |
| `pts_bench` | Bench points | Depth indicator |
| `ties` | Number of ties | Game flow |
| `leads` | Number of lead changes | Momentum |
| `poss_count` | Possession count | Pace |
| `lead_time` | Time with lead | Dominance |
| `large_lead` | Largest lead | Blowout indicator |
| `biggest_run` | Biggest scoring run | Momentum swing |

### 3.4 `<totals>/<statsbyprd>` - Per-Period Stats

**Path:** `/team/totals/statsbyprd[@prd='1']` and `[@prd='2']`

Used for:
- First half vs second half performance (momentum)
- Scoring trends
- Defensive adjustments

---

## Section 4: `<player>` - Player Statistics

**Path:** `/bbgame/team/player`

### Player Stats (per player):

| XML Attribute | Description |
|--------------|-------------|
| `uni` | Uniform number |
| `pno` | Player number |
| `code` | Player ID |
| `name` | Player name |
| `pos` | Position (G/F/C) |
| `gp` | Games played |
| `gs` | Games started |
| `min` | Minutes played |

### Player Box Score:

| XML Attribute | Description |
|--------------|-------------|
| `fgm`, `fga`, `fgpct` | Field goal stats |
| `fgm3`, `fga3`, `fg3pct` | 3-point stats |
| `ftm`, `fta`, `ftpct` | Free throw stats |
| `tp` | Total points |
| `oreb`, `dreb`, `treb` | Rebounds |
| `ast` | Assists |
| `stl` | Steals |
| `blk` | Blocks |
| `to` | Turnovers |
| `pf` | Personal fouls |
| `plusminus` | Plus/minus rating |
| `eff` | Efficiency rating |

---

## VAE Input Feature Mapping

### Team-Level Features (80 dimensions)

Based on the XML fields above, here's the recommended VAE input mapping:

```
Indices 0-19:  Shooting
  0-3:   fgm, fga, fgm3, fga3
  4-7:   ftm, fta, fgpct, fg3pct
  8-11:  ftpct, tp, pts_paint, pts_fastb
  12-19: (reserved for derived shooting %)

Indices 20-31: Rebounding  
  20-22: oreb, dreb, treb
  23-25: (offensive reb rate, def reb rate)
  26-31: (reserved)

Indices 32-39: Turnovers & Defense
  32-34: to, stl, blk
  35-39: (derived turnover rate, steal rate, block rate)

Indices 40-49: Playmaking
  40-42: ast, ast/turnover ratio
  43-49: (reserved for assist distribution)

Indices 50-59: Efficiency
  50-52: pts_to, pts_ch2, pts_bench
  53-59: (computed efficiency metrics)

Indices 60-69: Game Flow
  60-62: ties, leads, lead_time
  63-65: large_lead, biggest_run
  66-69: (pace indicators)

Indices 70-79: Opponent-Adjusted
  (Computed from both teams' stats for matchup features)
```

### Example: Computing Features from XML

```python
import xml.etree.ElementTree as ET

def parse_team_stats(xml_path: str, team_vh: str) -> dict:
    """Parse team stats from StatBroadcast XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    team = root.find(f".//team[@vh='{team_vh}']")
    stats = team.find(".//totals/stats")
    
    features = {
        'fgm': int(stats.get('fgm', 0)),
        'fga': int(stats.get('fga', 0)),
        'fgm3': int(stats.get('fgm3', 0)),
        'fga3': int(stats.get('fga3', 0)),
        'ftm': int(stats.get('ftm', 0)),
        'fta': int(stats.get('fta', 0)),
        'tp': int(stats.get('tp', 0)),
        'blk': int(stats.get('blk', 0)),
        'stl': int(stats.get('stl', 0)),
        'ast': int(stats.get('ast', 0)),
        'oreb': int(stats.get('oreb', 0)),
        'dreb': int(stats.get('dreb', 0)),
        'treb': int(stats.get('treb', 0)),
        'pf': int(stats.get('pf', 0)),
        'to': int(stats.get('to', 0)),
        'drawn': int(stats.get('drawn', 0)),
    }
    
    # Add special stats
    special = team.find(".//totals/special")
    if special is not None:
        features['pts_to'] = int(special.get('pts_to', 0))
        features['pts_paint'] = int(special.get('pts_paint', 0))
        features['pts_fastb'] = int(special.get('pts_fastb', 0))
        features['pts_bench'] = int(special.get('pts_bench', 0))
    
    return features
```

---

## Data Quality Notes

1. **Empty values:** Some fields may be empty strings ("") - handle with defaults
2. **Percentage fields:** May be strings like "50.0" or "N/A"
3. **Period structure:** Regular games have 2 halves, conference tournaments may have OT
4. **Incomplete games:** Check `status/@complete` before using data

---

## Latent Space Mapping (16 dimensions)

The VAE's latent representation (16-dim) should encode:

| Latent Dim | Description | Derived From |
|------------|-------------|--------------|
| 0-3 | Offensive capability | pts, fgpct, ast, pts_paint |
| 4-7 | Defensive capability | blk, stl, opponent to |
| 8-11 | Pace/style | poss_count, pts_fastb, pts_ch2 |
| 12-15 | Consistency | second_half vs first_half diff |

---

## Transition Probabilities (8 dimensions)

Decoder output represents:

| Index | Transition Type |
|-------|-----------------|
| 0 | Win → Win (momentum) |
| 1 | Win → Loss (letdown) |
| 2 | Loss → Win ( comeback) |
| 3 | Loss → Loss (collapse) |
| 4 | High scoring game |
| 5 | Low scoring game |
| 6 | High pace game |
| 7 | Low pace game |

---

## References

- Original schema: StatBroadcast proprietary
- steve.js integration: `/home/admin/repos/steve.js/tests/fixtures/statbroadcast-game-sample.xml`
- VAE feature extractor: steve.js `tests/sports/vae-feature-extractor.test.js`
