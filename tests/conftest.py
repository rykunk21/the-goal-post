"""
Pytest configuration and fixtures for NCAAB predictor tests.
"""
import os
import tempfile
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest


# ============================================================================
# Project Paths
# ============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def fixtures_dir(project_root: Path) -> Path:
    """Return the fixtures directory path."""
    return project_root / "tests" / "fixtures"


@pytest.fixture(scope="session")
def sample_xml_path(fixtures_dir: Path) -> Path:
    """Return the path to the sample XML file."""
    # Prefer the real StatBroadcast sample if available
    real_xml = fixtures_dir / "sample_game_16960.xml"
    if real_xml.exists():
        return real_xml
    return fixtures_dir / "sample_game.xml"


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_xml_data(sample_xml_path: Path) -> str:
    """
    Return sample StatBroadcast XML data for testing.
    
    If a fixture file exists, read it; otherwise return embedded sample.
    """
    if sample_xml_path.exists():
        return sample_xml_path.read_text()
    
    # Return embedded sample if file doesn't exist
    return SAMPLE_XML_GAME


@pytest.fixture
def sample_game_dict() -> Dict[str, Any]:
    """Return sample game data as a dictionary."""
    return {
        "game_id": "2024021501",
        "home_team": "Duke",
        "away_team": "North Carolina",
        "home_score": 78,
        "away_score": 72,
        "period": 2,
        "clock": "0:32",
        "home_possessions": 65,
        "away_possessions": 67,
        "home_turnovers": 8,
        "away_turnovers": 12,
        "home_offensive_rebounds": 10,
        "away_offensive_rebounds": 7,
        "home_defensive_rebounds": 22,
        "away_defensive_rebounds": 25,
    }


@pytest.fixture
def sample_team_stats() -> Dict[str, Any]:
    """Return sample team statistics."""
    return {
        "team_id": "duke",
        "team_name": "Duke Blue Devils",
        "conference": "ACC",
        "games_played": 25,
        "wins": 18,
        "losses": 7,
        "points_per_game": 78.5,
        "points_allowed": 68.2,
        "rebounds_per_game": 35.6,
        "assists_per_game": 15.2,
        "turnovers_per_game": 11.3,
        "steals_per_game": 7.1,
        "blocks_per_game": 3.4,
        "fg_pct": 47.8,
        "fg3_pct": 36.5,
        "ft_pct": 72.1,
    }


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_database():
    """Create a mock database connection."""
    mock_db = MagicMock()
    mock_db.execute = MagicMock(return_value=[])
    mock_db.fetchall = MagicMock(return_value=[])
    mock_db.fetchone = MagicMock(return_value=None)
    return mock_db


@pytest.fixture
def mock_api_response():
    """Create a mock API response."""
    return {
        "status": "success",
        "data": {
            "predictions": [
                {
                    "game_id": "2024021501",
                    "home_win_prob": 0.62,
                    "away_win_prob": 0.38,
                    "predicted_home_score": 76,
                    "predicted_away_score": 71,
                }
            ]
        },
    }


@pytest.fixture
def mock_model():
    """Create a mock ML model."""
    mock = MagicMock()
    mock.predict = MagicMock(return_value={
        "home_win_prob": 0.65,
        "away_win_prob": 0.35,
    })
    mock.predict_proba = MagicMock(return_value=[[0.35, 0.65]])
    return mock


# ============================================================================
# Temp Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_data_dir(temp_dir: Path) -> Path:
    """Create a temporary data directory."""
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def temp_model_dir(temp_dir: Path) -> Path:
    """Create a temporary model directory."""
    model_dir = temp_dir / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def temp_cache_dir(temp_dir: Path) -> Path:
    """Create a temporary cache directory."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir()
    return cache_dir


# ============================================================================
# Config Fixtures
# ============================================================================

@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Return mock configuration for testing."""
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_ncaab",
            "user": "test_user",
            "password": "test_pass",
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "debug": True,
        },
        "model": {
            "path": "/tmp/test_models",
            "input_dim": 50,
            "hidden_dim": 128,
            "output_dim": 2,
        },
        "cache": {
            "enabled": True,
            "ttl": 3600,
        },
    }


# ============================================================================
# Sample XML Data (embedded fallback)
# ============================================================================

SAMPLE_XML_GAME = """<?xml version="1.0" encoding="UTF-8"?>
<game>
    <gameId>2024021501</gameId>
    <date>2024-02-15</date>
    <time>19:00:00</time>
    <venue>
        <name>Cameron Indoor Stadium</name>
        <city>Durham</city>
        <state>NC</state>
    </venue>
    <homeTeam>
        <teamId>duke</teamId>
        <name>Duke Blue Devils</name>
        <conference>ACC</conference>
        <score>78</score>
        <periodScore>38</periodScore>
        <statistics>
            <fieldGoalsMade>28</fieldGoalsMade>
            <fieldGoalsAttempted>55</fieldGoalsAttempted>
            <threePointersMade>9</threePointersMade>
            <threePointersAttempted>22</threePointersAttempted>
            <freeThrowsMade>13</freeThrowsMade>
            <freeThrowsAttempted>18</freeThrowsAttempted>
            <reboundsOffensive>10</reboundsOffensive>
            <reboundsDefensive>22</reboundsDefensive>
            <reboundsTotal>32</reboundsTotal>
            <assists>15</assists>
            <turnovers>8</turnovers>
            <steals>6</steals>
            <blocks>3</blocks>
            <fouls>12</fouls>
        </statistics>
    </homeTeam>
    <awayTeam>
        <teamId>unc</teamId>
        <name>North Carolina Tar Heels</name>
        <conference>ACC</conference>
        <score>72</score>
        <periodScore>35</periodScore>
        <statistics>
            <fieldGoalsMade>26</fieldGoalsMade>
            <fieldGoalsAttempted>58</fieldGoalsAttempted>
            <threePointersMade>8</threePointersAttempted>
            <threePointersAttempted>21</threePointersAttempted>
            <freeThrowsMade>12</freeThrowsAttempted>
            <freeThrowsAttempted>16</freeThrowsAttempted>
            <reboundsOffensive>7</reboundsOffensive>
            <reboundsDefensive>25</reboundsDefensive>
            <reboundsTotal>32</reboundsTotal>
            <assists>12</assists>
            <turnovers>12</turnovers>
            <steals>5</steals>
            <blocks>2</blocks>
            <fouls>15</fouls>
        </statistics>
    </awayTeam>
    <period>2</period>
    <clock>0:32</clock>
    <status>final</status>
</game>
"""
