"""Explicit cfbfastR -> reviewed decision-segment adapter, regulation only.

College clock stamps are provider observations (often end-of-play), not claimed
NFL-equivalent snap times. This adapter validates extraction, not simulation.
"""
import numpy as np
import pandas as pd
from extraction import extract_game

PASS = {'Pass Completion','Pass Reception','Pass Incompletion','Passing Touchdown','Interception',
        'Interception Return','Interception Return Touchdown','Sack','Pass','Pass Interception Return'}
RUN = {'Rush','Rushing Touchdown'}
PUNT = {'Punt','Punt Return','Punt Return Touchdown','Blocked Punt',
        'Blocked Punt Touchdown','Punt Team Fumble Recovery'}
FG = {'Field Goal Good','Field Goal Missed','Blocked Field Goal',
      'Blocked Field Goal Touchdown','Missed Field Goal Return','Missed Field Goal Return Touchdown'}
OTHER_DECISION = {'Fumble','Fumble Recovery (Opponent)','Fumble Recovery (Own)',
                  'Fumble Return Touchdown','Fumble Recovery (Opponent) Touchdown',
                  'Fumble Recovery (Own) Touchdown','Safety','Penalty','Penalty Touchdown'}
NONDECISION = {'Kickoff','Kickoff Return (Offense)','Kickoff Return Touchdown',
               'Kickoff Team Fumble Recovery','Kickoff Return (Defense)',
               'End Period','End of Game','End of Half','End of Regulation','Timeout',
               'Extra Point Good','Extra Point Missed','Blocked PAT','PAT Good','PAT Missed',
               'Two Point Pass','Two Point Rush','Two Point Conversion','Defensive 2pt Conversion'}


def parsed_kind(kind, down):
    if kind in NONDECISION:
        return 'no_play', False
    mapping = [(PASS,'pass'),(RUN,'run'),(PUNT,'punt'),(FG,'field_goal'),(OTHER_DECISION,'no_play')]
    for names, result in mapping:
        if kind in names:
            # Down zero/no down is not an ordinary decision, but its rewards
            # remain in the full scoreboard sequence.
            return result, bool(pd.notna(down) and 1 <= down <= 4)
    raise ValueError(f'unreviewed college play type: {kind}')


def college_frame(raw, metadata):
    g=raw.sort_values('game_row_number',kind='stable').reset_index(drop=True)
    if g.game_row_number.isna().any() or g.game_row_number.duplicated().any():
        raise ValueError('missing/duplicate source row order')
    home_values=g.home.dropna().unique(); away_values=g.away.dropna().unique()
    if len(home_values)!=1 or len(away_values)!=1 or home_values[0]==away_values[0]:
        raise ValueError('ambiguous source home/away')
    home,away=str(home_values[0]),str(away_values[0])
    # Require an exact numeric identity whenever the source supplies it.
    # Otherwise require exact provider team-name agreement with the schedule.
    for side, source_name in [('home',home),('away',away)]:
        ids=g[side+'_team_id'].dropna().unique()
        if len(ids)>1 or (len(ids)==1 and str(int(ids[0])) != metadata[side+'_team_id']):
            raise ValueError(f'{side} source/schedule identity mismatch')
        if len(ids)==0 and source_name != metadata[side+'_team']:
            raise ValueError(f'{side} source/schedule name mismatch')
    scores=[]
    for r in g.itertuples():
        if r.pos_team not in (home,away) or r.def_pos_team not in (home,away) or r.pos_team==r.def_pos_team:
            raise ValueError('invalid college possession identity')
        p,d=r.pos_team_score,r.def_pos_team_score
        scores.append((p,d) if r.pos_team==home else (d,p))
    scores=np.asarray(scores,dtype=float)
    if not np.isfinite(scores).all() or np.any(scores<0) or np.any(scores!=scores.astype(int)):
        raise ValueError('invalid college post-play scoreboard')
    if np.any(np.diff(np.vstack(([0,0],scores)),axis=0)<0):
        raise ValueError('college scoreboard reversal')
    expected=(metadata['home_final_score'],metadata['away_final_score'])
    if tuple(scores[-1].astype(int))!=expected:
        raise ValueError('college play-by-play/schedule final mismatch')
    pre=np.vstack(([0,0],scores[:-1]))
    home_pos=(g.pos_team==home).to_numpy()
    decisions=[parsed_kind(r.play_type,r.down) for r in g.itertuples()]
    frame=pd.DataFrame(dict(
        game_id=metadata['game_id'],week=metadata['week'],game_date=metadata['game_date'],
        home_team=home,away_team=away,home_score=expected[0],away_score=expected[1],
        play_id=g.game_row_number.to_numpy(),qtr=g.period.to_numpy(),
        down=[r.down if decision else np.nan for r,(_,decision) in zip(g.itertuples(),decisions)],
        ydstogo=g.distance.to_numpy(),yardline_100=g.yards_to_goal.to_numpy(),
        play_type=[kind for kind,_ in decisions],posteam=g.pos_team.to_numpy(),
        half_seconds_remaining=g.TimeSecsRem.to_numpy(),
        total_home_score=scores[:,0],total_away_score=scores[:,1],
        posteam_score_post=np.where(home_pos,scores[:,0],scores[:,1]),
        defteam_score_post=np.where(home_pos,scores[:,1],scores[:,0]),
        posteam_score=np.where(home_pos,pre[:,0],pre[:,1]),
        defteam_score=np.where(home_pos,pre[:,1],pre[:,0]),
        extra_point_attempt=0,two_point_attempt=0))
    return frame


def extract_college(raw, metadata):
    return extract_game(college_frame(raw,metadata))
