"""Football-only transition extraction, chronological imputation and simulation.

The previous NFLTransitionModel terminal-outcome API has been replaced.
See the root README for versioned data and package-module commands.
"""
from .extraction import extract_game
from .build_reset import fill, History
from .read_game_rows import read, unpack, team_view
from .simulator import run
__all__ = ["extract_game", "fill", "History", "read", "unpack", "team_view", "run"]
