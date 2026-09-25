"""Whole-game fallback using ESPN-derived raw team, score and order columns."""
import numpy as np
import pandas as pd
from college_adapter import parsed_kind
from extraction import extract_game


def extract_espn_college(raw,meta):
    g=raw.sort_values('game_play_number',kind='stable').reset_index(drop=True)
    if g.game_play_number.isna().any() or g.game_play_number.duplicated().any():
        raise ValueError('ESPN missing/duplicate game play order')
    if not (g.status_type_completed==True).all():raise ValueError('ESPN game incomplete')
    for side in ['home','away']:
        vals=g[side+'TeamId'].dropna().unique()
        if len(vals)!=1 or str(int(vals[0]))!=meta[side+'_team_id']:
            raise ValueError('ESPN schedule/team identity mismatch')
        vals=g[side+'FinalScore'].dropna().unique()
        if len(vals)!=1 or int(vals[0])!=meta[side+'_final_score']:
            raise ValueError('ESPN schedule/final mismatch')
    scores=g[['homeScore','awayScore']].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or np.any(scores<0) or np.any(scores!=scores.astype(int)):
        raise ValueError('ESPN invalid scoreboard')
    if np.any(np.diff(np.vstack(([0,0],scores)),axis=0)<0):raise ValueError('ESPN scoreboard reversal')
    if tuple(scores[-1].astype(int))!=(meta['home_final_score'],meta['away_final_score']):
        raise ValueError('ESPN last play/final mismatch')
    periods=g['period.number'].to_numpy()
    if not np.isfinite(periods).all() or np.any(periods<1) or np.any(periods!=periods.astype(int)):
        raise ValueError('ESPN invalid period')
    if np.any(np.diff(periods)<0):raise ValueError('ESPN period order reversal')
    pre=np.vstack(([0,0],scores[:-1]));records=[]
    home,away=meta['home_team'],meta['away_team']
    for i,r in g.iterrows():
        kind,decision=parsed_kind(r['type.text'],r['start.down'])
        if r['period.number']>4:decision=False
        team=str(int(r['start.team.id'])) if pd.notna(r['start.team.id']) else None
        if decision and team not in (meta['home_team_id'],meta['away_team_id']):
            raise ValueError('ESPN decision possession missing/unknown')
        home_pos=team!=meta['away_team_id']
        clock=str(r['clock.displayValue']).split(':')
        if len(clock)!=2:raise ValueError('ESPN missing clock')
        mins,secs=map(int,clock)
        if not 0<=mins<=15 or not 0<=secs<60 or 60*mins+secs>900:raise ValueError('ESPN invalid clock')
        remaining=(900 if int(r['period.number']) in (1,3) else 0)+60*mins+secs
        records.append(dict(game_id=meta['game_id'],week=meta['week'],game_date=meta['game_date'],
            home_team=home,away_team=away,home_score=meta['home_final_score'],away_score=meta['away_final_score'],
            play_id=int(r.game_play_number),qtr=r['period.number'],play_type=kind,
            down=r['start.down'] if decision else np.nan,ydstogo=r['start.distance'],
            yardline_100=r['start.yardsToEndzone'],posteam=home if home_pos else away,
            half_seconds_remaining=remaining,extra_point_attempt=0,two_point_attempt=0,
            total_home_score=scores[i,0],total_away_score=scores[i,1],
            posteam_score_post=scores[i,0 if home_pos else 1],defteam_score_post=scores[i,1 if home_pos else 0],
            posteam_score=pre[i,0 if home_pos else 1],defteam_score=pre[i,1 if home_pos else 0]))
    return extract_game(pd.DataFrame(records))
