"""
Feature Extraction Functions for NCAAB Prediction

Provides individual functions to compute specific basketball metrics.
Each function calculates exactly one metric.
"""

from typing import Union, List, Dict, Any, Optional
import math


def calculate_points_per_possession(points: Union[int, float], 
                                     possessions: Union[int, float]) -> float:
    """
    Calculate points scored per possession (PPP).
    
    Args:
        points: Total points scored
        possessions: Number of possessions
        
    Returns:
        Points per possession (higher is better offensively)
        
    Handles:
        - Division by zero (returns 0.0)
        - Missing/invalid data (returns 0.0)
    """
    # Handle missing or invalid data
    if points is None or possessions is None:
        return 0.0
    
    try:
        points = float(points)
        possessions = float(possessions)
    except (TypeError, ValueError):
        return 0.0
    
    # Handle division by zero
    if possessions == 0:
        return 0.0
    
    return points / possessions


def calculate_effective_fg_percent(fgm: Union[int, float], 
                                    fg3m: Union[int, float], 
                                    fga: Union[int, float]) -> float:
    """
    Calculate effective field goal percentage (eFG%).
    
    Accounts for three-pointers being worth more than two-pointers.
    Formula: (FGM + 0.5 * 3PM) / FGA
    
    Args:
        fgm: Field goals made
        fg3m: Three-point field goals made
        fga: Field goals attempted
        
    Returns:
        Effective field goal percentage (0.0 to 1.5, typically 0.0 to ~0.6)
        
    Handles:
        - Division by zero (returns 0.0)
        - Missing/invalid data (returns 0.0)
    """
    # Handle missing or invalid data
    if any(x is None for x in [fgm, fg3m, fga]):
        return 0.0
    
    try:
        fgm = float(fgm)
        fg3m = float(fg3m)
        fga = float(fga)
    except (TypeError, ValueError):
        return 0.0
    
    # Handle division by zero
    if fga == 0:
        return 0.0
    
    return (fgm + 0.5 * fg3m) / fga


def calculate_turnover_rate(turnovers: Union[int, float], 
                             possessions: Union[int, float]) -> float:
    """
    Calculate turnover rate (percentage of possessions that end in turnovers).
    
    Args:
        turnovers: Number of turnovers
        possessions: Number of possessions
        
    Returns:
        Turnover rate as decimal (0.0 to 1.0, lower is better)
        
    Handles:
        - Division by zero (returns 0.0)
        - Missing/invalid data (returns 0.0)
    """
    # Handle missing or invalid data
    if turnovers is None or possessions is None:
        return 0.0
    
    try:
        turnovers = float(turnovers)
        possessions = float(possessions)
    except (TypeError, ValueError):
        return 0.0
    
    # Handle division by zero
    if possessions == 0:
        return 0.0
    
    return turnovers / possessions


def calculate_offensive_rebound_rate(oreb: Union[int, float], 
                                      total_reb: Union[int, float]) -> float:
    """
    Calculate offensive rebound rate (percentage of missed shots rebound by offense).
    
    Also known as OREB%.
    
    Args:
        oreb: Offensive rebounds
        total_reb: Total rebounds (offensive + defensive)
        
    Returns:
        Offensive rebound rate as decimal (0.0 to 1.0)
        
    Handles:
        - Division by zero (returns 0.0)
        - Missing/invalid data (returns 0.0)
    """
    # Handle missing or invalid data
    if oreb is None or total_reb is None:
        return 0.0
    
    try:
        oreb = float(oreb)
        total_reb = float(total_reb)
    except (TypeError, ValueError):
        return 0.0
    
    # Handle division by zero
    if total_reb == 0:
        return 0.0
    
    return oreb / total_reb


def calculate_strength_of_schedule(record_list: List[Dict[str, Any]]) -> float:
    """
    Calculate strength of schedule based on team records.
    
    Uses opponent win percentage as a simple SOS metric.
    
    Args:
        record_list: List of dicts with 'wins' and 'losses' keys, 
                     or list of [wins, losses] tuples
                     
    Returns:
        Average opponent win percentage (0.0 to 1.0)
        
    Handles:
        - Empty list (returns 0.5 - neutral)
        - Missing/invalid data (skips invalid entries)
    """
    # Handle empty or invalid input
    if not record_list:
        return 0.5  # Neutral SOS for no data
    
    total_win_pct = 0.0
    valid_games = 0
    
    for record in record_list:
        try:
            # Support dict format: {'wins': X, 'losses': Y}
            if isinstance(record, dict):
                wins = record.get('wins', 0) or 0
                losses = record.get('losses', 0) or 0
            # Support tuple/list format: [wins, losses]
            elif isinstance(record, (list, tuple)) and len(record) >= 2:
                wins = float(record[0]) if record[0] is not None else 0
                losses = float(record[1]) if record[1] is not None else 0
            else:
                continue
                
            total_games = wins + losses
            
            # Skip if no games played
            if total_games == 0:
                continue
                
            total_win_pct += wins / total_games
            valid_games += 1
            
        except (TypeError, ValueError, KeyError):
            # Skip invalid entries
            continue
    
    # Return neutral if no valid games found
    if valid_games == 0:
        return 0.5
    
    return total_win_pct / valid_games


# Additional useful feature functions for NCAAB prediction

def calculate_true_shooting_percent(points: Union[int, float],
                                    fga: Union[int, float],
                                    fta: Union[int, float]) -> float:
    """
    Calculate true shooting percentage (TS%).
    
    Accounts for free throws and shooting efficiency.
    Formula: Points / (2 * (FGA + 0.44 * FTA))
    
    Args:
        points: Total points scored
        fga: Field goals attempted
        fta: Free throws attempted
        
    Returns:
        True shooting percentage (0.0 to ~0.7 typically)
    """
    if any(x is None for x in [points, fga, fta]):
        return 0.0
    
    try:
        points = float(points)
        fga = float(fga)
        fta = float(fta)
    except (TypeError, ValueError):
        return 0.0
    
    denominator = 2 * (fga + 0.44 * fta)
    if denominator == 0:
        return 0.0
    
    return points / denominator


def calculate_assist_to_turnover_ratio(assists: Union[int, float],
                                      turnovers: Union[int, float]) -> float:
    """
    Calculate assist to turnover ratio.
    
    Args:
        assists: Number of assists
        turnovers: Number of turnovers
        
    Returns:
        Assist-to-turnover ratio (higher is better)
    """
    if assists is None or turnovers is None:
        return 0.0
    
    try:
        assists = float(assists)
        turnovers = float(turnovers)
    except (TypeError, ValueError):
        return 0.0
    
    if turnovers == 0:
        return assists  # Perfect if no turnovers
    
    return assists / turnovers


def calculate_block_plus_steal_rate(blocks: Union[int, float],
                                     steals: Union[int, float],
                                     possessions: Union[int, float]) -> float:
    """
    Calculate combined block and steal rate per possession.
    
    Args:
        blocks: Number of blocks
        steals: Number of steals
        possessions: Number of possessions
        
    Returns:
        Combined block + steal rate per possession
    """
    if any(x is None for x in [blocks, steals, possessions]):
        return 0.0
    
    try:
        blocks = float(blocks)
        steals = float(steals)
        possessions = float(possessions)
    except (TypeError, ValueError):
        return 0.0
    
    if possessions == 0:
        return 0.0
    
    return (blocks + steals) / possessions


def calculate_pace(possessions: float, minutes: float = 40.0) -> float:
    """
    Calculate pace (possessions per game, normalized to 40 minutes).
    
    Args:
        possessions: Number of possessions
        minutes: Game length in minutes (default 40 for college)
        
    Returns:
        Pace rating (possessions per 40 minutes)
    """
    if possessions is None or minutes is None:
        return 0.0
    
    try:
        possessions = float(possessions)
        minutes = float(minutes)
    except (TypeError, ValueError):
        return 0.0
    
    if minutes == 0:
        return 0.0
    
    return (possessions / minutes) * 40.0


def calculate_win_percentage(wins: Union[int, float],
                              losses: Union[int, float]) -> float:
    """
    Calculate win percentage.
    
    Args:
        wins: Number of wins
        losses: Number of losses
        
    Returns:
        Win percentage (0.0 to 1.0)
    """
    if wins is None or losses is None:
        return 0.0
    
    try:
        wins = float(wins)
        losses = float(losses)
    except (TypeError, ValueError):
        return 0.0
    
    total = wins + losses
    if total == 0:
        return 0.0
    
    return wins / total


def calculate_rebound_margin(oreb: Union[int, float],
                             dreb: Union[int, float],
                             opp_oreb: Union[int, float],
                             opp_dreb: Union[int, float]) -> float:
    """
    Calculate rebound margin (team rebounds - opponent rebounds).
    
    Args:
        oreb: Offensive rebounds
        dreb: Defensive rebounds
        opp_oreb: Opponent offensive rebounds
        opp_dreb: Opponent defensive rebounds
        
    Returns:
        Rebound margin (positive is better)
    """
    team_reb = (oreb or 0) + (dreb or 0)
    opp_reb = (opp_oreb or 0) + (opp_dreb or 0)
    
    return team_reb - opp_reb
