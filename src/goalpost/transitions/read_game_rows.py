"""Read one-game rows and restore the versioned probability tensor."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def read(path):
    frame=pd.read_parquet(path)
    if frame.game_id.duplicated().any():raise ValueError('More than one row per game')
    return frame


def unpack(row):
    if row['matrix_status']!='parsed_regulation':
        raise ValueError(f"No usable matrix: {row['matrix_status']}: {row['matrix_issue']}")
    shape=tuple(row['matrix_shape'])
    return np.asarray(row['transition_probabilities_flat'],dtype=np.float32).reshape(shape)


def team_view(row, team_id):
    """A team's offense and defense observations; never discard the opponent."""
    probabilities=unpack(row)
    if str(team_id)==row['home_team_id']:
        return probabilities
    if str(team_id)==row['away_team_id']:
        return probabilities[::-1].copy()
    raise ValueError('Team is not in this game')
