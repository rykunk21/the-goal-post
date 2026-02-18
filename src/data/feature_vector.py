"""
Feature Vector Assembly for NCAAB Prediction

Builds an 80-dimensional feature vector from team statistics.
Each dimension represents a specific predictive feature.
"""

from typing import Dict, List, Union, Optional
import numpy as np

from .feature_extraction import (
    calculate_points_per_possession,
    calculate_effective_fg_percent,
    calculate_turnover_rate,
    calculate_offensive_rebound_rate,
    calculate_strength_of_schedule,
    calculate_true_shooting_percent,
    calculate_assist_to_turnover_ratio,
    calculate_block_plus_steal_rate,
    calculate_pace,
    calculate_win_percentage,
    calculate_rebound_margin,
)


# Feature dimension constants (for indexing)
FEATURE_NAMES = [
    # Offensive Efficiency (0-7)
    'off_ppp',           # 0: Points per possession
    'off_efg',           # 1: Effective FG%
    'off_ts',            # 2: True shooting %
    'off_2p_pct',        # 3: 2-point FG%
    'off_3p_pct',        # 4: 3-point FG%
    'off_ft_pct',        # 5: Free throw %
    'off_ast_to',        # 6: Assist to turnover ratio
    'off_orb_rate',      # 7: Offensive rebound rate
    
    # Defensive Efficiency (8-15)
    'def_ppp',           # 8: Opponent points per possession
    'def_efg',           # 9: Opponent effective FG%
    'def_ts',            # 10: Opponent true shooting %
    'def_2p_pct',        # 11: Opponent 2-point FG%
    'def_3p_pct',        # 12: Opponent 3-point FG%
    'def_ft_pct',        # 13: Opponent free throw %
    'def_to_rate',       # 14: Turnover rate forced
    'def_drb_rate',      # 15: Defensive rebound rate
    
    # Rebounding (16-23)
    'off_orb',           # 16: Offensive rebounds per game
    'off_drb',           # 17: Defensive rebounds per game
    'off_total_rb',     # 18: Total rebounds per game
    'def_orb',           # 19: Opponent offensive rebounds allowed
    'def_drb',           # 20: Opponent defensive rebounds allowed
    'def_total_rb',     # 21: Total rebounds allowed per game
    'rb_margin',         # 22: Rebound margin
    'rb_rate_diff',     # 23: Rebound rate advantage
    
    # Turnovers & Possession (24-31)
    'off_to',           # 24: Turnovers per game
    'def_to',           # 25: Steals per game (forced turnovers)
    'to_margin',        # 26: Turnover margin
    'off_poss',         # 27: Possessions per game
    'def_poss',         # 28: Opponent possessions
    'pace',             # 29: Pace (possessions per 40 min)
    'to_rate',          # 30: Offensive turnover rate
    'steal_rate',       # 31: Steal rate
    
    # Shooting Volume (32-39)
    'off_fga',          # 32: Field goal attempts per game
    'off_fgm',          # 33: Field goals made per game
    'off_fg3a',         # 34: 3-point attempts per game
    'off_fg3m',         # 35: 3-pointers made per game
    'off_fta',          # 36: Free throw attempts per game
    'off_ftm',          # 37: Free throws made per game
    'off_2pa',          # 38: 2-point attempts per game
    'off_2pm',          # 39: 2-pointers made per game
    
    # Blocks & Steals (40-47)
    'off_blk',          # 40: Blocks per game (own)
    'off_stl',          # 41: Steals per game (own)
    'def_blk',          # 42: Blocks allowed per game
    'def_stl',          # 43: Steals allowed per game
    'blk_stl_rate',     # 44: Block + steal rate
    'def_blk_rate',     # 45: Block rate allowed
    'def_stl_rate',     # 46: Steal rate forced
    'blk_stl_margin',   # 47: Block + steal margin
    
    # Assists & Fouls (48-55)
    'off_ast',          # 48: Assists per game
    'off_fouls',        # 49: Personal fouls per game
    'def_ast',          # 50: Assists allowed
    'def_fouls',        # 51: Fouls drawn (opponent fouls)
    'def_ft_rate',      # 52: Free throw rate defense
    'off_ft_rate',      # 53: Free throw rate offense
    'ast_ratio',        # 54: Assist to FGM ratio
    'foul_margin',      # 55: Foul margin
    
    # Scoring Distribution (56-63)
    'off_ppg',          # 56: Points per game
    'def_ppg',          # 57: Points allowed per game
    'scoring_margin',   # 58: Scoring margin
    'off_1h_pts',       # 59: First half points
    'off_2h_pts',       # 60: Second half points
    'off_ot_pts',       # 61: Overtime points
    'close_games',      # 62: Close game win %
    'away_games',       # 63: Away game win %
    
    # Season/Team Context (64-71)
    'win_pct',          # 64: Overall win percentage
    'sos',              # 65: Strength of schedule
    'last_10_w_pct',    # 66: Win % in last 10 games
    'last_5_w_pct',    # 67: Win % in last 5 games
    'conf_win_pct',    # 68: Conference win %
    'rest_days',        # 69: Days of rest
    'is_home',          # 70: Home game indicator
    'tournament',       # 71: Tournament game indicator
    
    # Advanced Metrics (72-79)
    'off_rating',       # 72: Offensive rating (adjusted)
    'def_rating',      # 73: Defensive rating (adjusted)
    'net_rating',      # 74: Net rating (off - def)
    'pace_adj_off',    # 75: Pace-adjusted offensive rating
    'pace_adj_def',    # 76: Pace-adjusted defensive rating
    'Four_Factor_efg',  # 77: Four factors eFG%
    'Four_Factor_to',   # 78: Four factors turnover %
    'Four_Factor_orb',  # 79: Four factors OREB%
]


def _safe_float(value: any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_feature_vector(team_stats: dict, opponent_stats: Optional[dict] = None) -> np.ndarray:
    """
    Build an 80-dimensional feature vector from team statistics.
    
    Args:
        team_stats: Dictionary containing the team's statistics
        opponent_stats: Optional dictionary containing opponent's statistics
        
    Returns:
        numpy array of shape (80,) containing feature values
    """
    features = np.zeros(80, dtype=np.float64)
    
    # Default opponent stats to empty dict if not provided
    opp = opponent_stats if opponent_stats else {}
    team = team_stats if team_stats else {}
    
    # Helper to extract stats with defaults
    def getStat(stats: dict, key: str, default: float = 0.0) -> float:
        return _safe_float(stats.get(key), default)
    
    # ===== OFFENSIVE EFFICIENCY (0-7) =====
    features[0] = calculate_points_per_possession(
        getStat(team, 'points'),
        getStat(team, 'possessions')
    )
    features[1] = calculate_effective_fg_percent(
        getStat(team, 'fgm'),
        getStat(team, 'fg3m'),
        getStat(team, 'fga')
    )
    features[2] = calculate_true_shooting_percent(
        getStat(team, 'points'),
        getStat(team, 'fga'),
        getStat(team, 'fta')
    )
    # 2-point percentage
    fgm = getStat(team, 'fgm')
    fg3m = getStat(team, 'fg3m')
    fga = getStat(team, 'fga')
    fg2a = fga - getStat(team, 'fg3a')
    fg2m = fgm - fg3m
    features[3] = fg2m / fg2a if fg2a > 0 else 0.0
    # 3-point percentage
    features[4] = fg3m / getStat(team, 'fg3a') if getStat(team, 'fg3a') > 0 else 0.0
    # Free throw percentage
    features[5] = getStat(team, 'ftm') / getStat(team, 'fta') if getStat(team, 'fta') > 0 else 0.0
    # Assist to turnover ratio
    features[6] = calculate_assist_to_turnover_ratio(
        getStat(team, 'assists'),
        getStat(team, 'turnovers')
    )
    # Offensive rebound rate
    features[7] = calculate_offensive_rebound_rate(
        getStat(team, 'oreb'),
        getStat(team, 'rebounds')
    )
    
    # ===== DEFENSIVE EFFICIENCY (8-15) =====
    features[8] = calculate_points_per_possession(
        getStat(opp, 'points'),
        getStat(opp, 'possessions')
    )
    features[9] = calculate_effective_fg_percent(
        getStat(opp, 'fgm'),
        getStat(opp, 'fg3m'),
        getStat(opp, 'fga')
    )
    features[10] = calculate_true_shooting_percent(
        getStat(opp, 'points'),
        getStat(opp, 'fga'),
        getStat(opp, 'fta')
    )
    # Opponent 2-point percentage
    opp_fgm = getStat(opp, 'fgm')
    opp_fg3m = getStat(opp, 'fg3m')
    opp_fga = getStat(opp, 'fga')
    opp_fg2a = opp_fga - getStat(opp, 'fg3a')
    opp_fg2m = opp_fgm - opp_fg3m
    features[11] = opp_fg2m / opp_fg2a if opp_fg2a > 0 else 0.0
    # Opponent 3-point percentage
    features[12] = opp_fg3m / getStat(opp, 'fg3a') if getStat(opp, 'fg3a') > 0 else 0.0
    # Opponent free throw percentage
    features[13] = getStat(opp, 'ftm') / getStat(opp, 'fta') if getStat(opp, 'fta') > 0 else 0.0
    # Turnover rate forced
    features[14] = calculate_turnover_rate(
        getStat(opp, 'turnovers'),
        getStat(opp, 'possessions')
    )
    # Defensive rebound rate
    features[15] = 1.0 - calculate_offensive_rebound_rate(
        getStat(opp, 'oreb'),
        getStat(opp, 'rebounds')
    ) if getStat(opp, 'rebounds') > 0 else 0.0
    
    # ===== REBOUNDING (16-23) =====
    features[16] = getStat(team, 'oreb')
    features[17] = getStat(team, 'dreb')
    features[18] = getStat(team, 'rebounds')
    features[19] = getStat(opp, 'oreb')
    features[20] = getStat(opp, 'dreb')
    features[21] = getStat(opp, 'rebounds')
    features[22] = calculate_rebound_margin(
        getStat(team, 'oreb'),
        getStat(team, 'dreb'),
        getStat(opp, 'oreb'),
        getStat(opp, 'dreb')
    )
    # Rebound rate difference
    team_orb_rate = features[7]  # Already computed
    opp_orb_rate = calculate_offensive_rebound_rate(
        getStat(opp, 'oreb'),
        getStat(opp, 'rebounds')
    )
    features[23] = team_orb_rate - opp_orb_rate
    
    # ===== TURNOVERS & POSSESSION (24-31) =====
    features[24] = getStat(team, 'turnovers')
    features[25] = getStat(team, 'steals')
    features[26] = getStat(team, 'steals') - getStat(team, 'turnovers')
    features[27] = getStat(team, 'possessions')
    features[28] = getStat(opp, 'possessions')
    features[29] = calculate_pace(getStat(team, 'possessions'))
    features[30] = calculate_turnover_rate(
        getStat(team, 'turnovers'),
        getStat(team, 'possessions')
    )
    features[31] = getStat(team, 'steals') / getStat(team, 'possessions') if getStat(team, 'possessions') > 0 else 0.0
    
    # ===== SHOOTING VOLUME (32-39) =====
    features[32] = getStat(team, 'fga')
    features[33] = getStat(team, 'fgm')
    features[34] = getStat(team, 'fg3a')
    features[35] = getStat(team, 'fg3m')
    features[36] = getStat(team, 'fta')
    features[37] = getStat(team, 'ftm')
    features[38] = fg2a
    features[39] = fg2m
    
    # ===== BLOCKS & STEALS (40-47) =====
    features[40] = getStat(team, 'blocks')
    features[41] = getStat(team, 'steals')
    features[42] = getStat(opp, 'blocks')
    features[43] = getStat(opp, 'steals')
    features[44] = calculate_block_plus_steal_rate(
        getStat(team, 'blocks'),
        getStat(team, 'steals'),
        getStat(team, 'possessions')
    )
    features[45] = getStat(opp, 'blocks') / getStat(team, 'possessions') if getStat(team, 'possessions') > 0 else 0.0
    features[46] = getStat(team, 'steals') / getStat(opp, 'possessions') if getStat(opp, 'possessions') > 0 else 0.0
    features[47] = (getStat(team, 'blocks') + getStat(team, 'steals')) - (getStat(opp, 'blocks') + getStat(opp, 'steals'))
    
    # ===== ASSISTS & FOULS (48-55) =====
    features[48] = getStat(team, 'assists')
    features[49] = getStat(team, 'fouls')
    features[50] = getStat(opp, 'assists')
    features[51] = getStat(opp, 'fouls')
    # Free throw rate (FTA / FGA)
    features[52] = getStat(opp, 'fta') / getStat(opp, 'fga') if getStat(opp, 'fga') > 0 else 0.0
    features[53] = getStat(team, 'fta') / getStat(team, 'fga') if getStat(team, 'fga') > 0 else 0.0
    # Assist to FGM ratio
    features[54] = getStat(team, 'assists') / getStat(team, 'fgm') if getStat(team, 'fgm') > 0 else 0.0
    features[55] = getStat(opp, 'fouls') - getStat(team, 'fouls')
    
    # ===== SCORING DISTRIBUTION (56-63) =====
    features[56] = getStat(team, 'points')
    features[57] = getStat(opp, 'points')
    features[58] = getStat(team, 'points') - getStat(opp, 'points')
    features[59] = getStat(team, 'first_half_pts', getStat(team, 'half1_pts', getStat(team, '1h_pts', 0)))
    features[60] = getStat(team, 'second_half_pts', getStat(team, 'half2_pts', getStat(team, '2h_pts', 0)))
    features[61] = getStat(team, 'overtime_pts', getStat(team, 'ot_pts', 0))
    # Close games and away games - use season context if available
    features[62] = getStat(team, 'close_game_win_pct', getStat(team, 'close_win_pct', 0.5))
    features[63] = getStat(team, 'away_win_pct', getStat(team, 'road_win_pct', 0.5))
    
    # ===== SEASON/TEAM CONTEXT (64-71) =====
    features[64] = calculate_win_percentage(
        getStat(team, 'wins'),
        getStat(team, 'losses')
    )
    features[65] = getStat(team, 'sos', getStat(team, 'strength_of_schedule', 0.5))
    features[66] = getStat(team, 'last_10_win_pct', getStat(team, 'last10_pct', 0.5))
    features[67] = getStat(team, 'last_5_win_pct', getStat(team, 'last5_pct', 0.5))
    features[68] = getStat(team, 'conference_win_pct', getStat(team, 'conf_win_pct', 0.5))
    features[69] = getStat(team, 'rest_days', 2.0)  # Default to 2 days rest
    features[70] = 1.0 if getStat(team, 'home_game', 0) == 1 else 0.0
    features[71] = 1.0 if getStat(team, 'tournament', getStat(team, 'ncaa_tournament', 0)) == 1 else 0.0
    
    # ===== ADVANCED METRICS (72-79) =====
    # Offensive rating (points per 100 possessions)
    features[72] = features[0] * 100.0 if features[0] > 0 else 0.0
    # Defensive rating
    features[73] = features[8] * 100.0 if features[8] > 0 else 0.0
    # Net rating
    features[74] = features[72] - features[73]
    # Pace-adjusted offensive rating
    avg_pace = 70.0  # Average college pace
    features[75] = features[72] * (avg_pace / features[29]) if features[29] > 0 else features[72]
    # Pace-adjusted defensive rating
    features[76] = features[73] * (avg_pace / features[29]) if features[29] > 0 else features[73]
    # Four factors - eFG%
    features[77] = features[1]
    # Four factors - TO%
    features[78] = features[30]
    # Four factors - OREB%
    features[79] = features[7]
    
    return features


def get_feature_names() -> List[str]:
    """
    Return list of feature names for the 80-dimensional vector.
    
    Returns:
        List of 80 feature name strings
    """
    return FEATURE_NAMES.copy()


def get_feature_index(feature_name: str) -> int:
    """
    Get the index of a named feature.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        Index into the feature vector
        
    Raises:
        ValueError: If feature_name is not found
    """
    if feature_name not in FEATURE_NAMES:
        raise ValueError(f"Unknown feature: {feature_name}")
    return FEATURE_NAMES.index(feature_name)
