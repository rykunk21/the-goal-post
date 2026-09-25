import copy
import numpy as np
import pandas as pd
import pytest
from college_adapter import college_frame,extract_college,parsed_kind
from build_tables import matrix


def fixture():
    rows=[]
    spec=[('Kickoff',1,1800,'H',0,0,1),('Rush',1,1790,'H',0,0,1),
          ('Passing Touchdown',1,1760,'H',7,0,2),('Timeout',1,1760,'H',7,0,2),
          ('Kickoff',1,1760,'A',0,7,1),('Punt',1,1700,'A',0,7,4),
          ('End of Half',2,0,'A',0,7,0),('Kickoff',3,1800,'A',0,7,1),
          ('Rush',3,1750,'A',0,7,1),('Interception Return Touchdown',3,1740,'A',0,14,2),
          ('End of Game',4,0,'A',0,14,0)]
    for i,(kind,period,clock,pos,own,opp,down) in enumerate(spec,1):
        rows.append(dict(game_id=1,game_row_number=i,id_play=str(i),home='H',away='A',
            home_team_id=10,away_team_id=20,pos_team=pos,def_pos_team='A' if pos=='H' else 'H',
            pos_team_score=own,def_pos_team_score=opp,period=period,TimeSecsRem=clock,
            down=down,distance=10,yards_to_goal=50,play_type=kind))
    meta=dict(game_id='1',week=1,game_date='2026-09-05',home_team='H',away_team='A',
              home_team_id='10',away_team_id='20',home_final_score=14,away_final_score=0)
    return pd.DataFrame(rows),meta


def test_decisions_not_fake_kickoff_or_timeout_downs():
    raw,meta=fixture();g=extract_college(raw,meta)
    assert [int(e.play_id) for e in g.events]==[2,3,6,9,10]
    assert any(e.play_type=='punt' and e.source//18==3 for e in g.events)
    assert g.regulation==(14,0)
    assert any(e.opponent_points==7 and not e.home_offense for e in g.events)


def test_flat_probabilities_counts_and_masks():
    raw,meta=fixture();g=extract_college(raw,meta)
    keys=sorted({e.key for e in g.events})
    p,c,t=matrix(g,keys)
    assert p.ndim==c.ndim==t.ndim==1
    assert len(p)==len(c)==2*9*len(keys)
    assert c.sum()==t.sum()==len(g.events)
    p=p.reshape(2,9,len(keys));t=t.reshape(2,9,72)
    for source in set(k[0] for k in keys):
        ix=[i for i,k in enumerate(keys) if k[0]==source]
        assert np.allclose(p[:,:,ix].sum(-1),(t[:,:,source]>0).astype(float))


def test_reversal_not_silently_clipped():
    raw,meta=fixture();raw.loc[3,'pos_team_score']=0
    with pytest.raises(ValueError,match='reversal'):college_frame(raw,meta)


def test_external_final_mismatch_fails():
    raw,meta=fixture();meta['home_final_score']=17
    with pytest.raises(ValueError,match='final mismatch'):college_frame(raw,meta)


def test_home_away_identity_mismatch_fails():
    raw,meta=fixture();meta['home_team_id']='20'
    with pytest.raises(ValueError,match='identity mismatch'):college_frame(raw,meta)


def test_unknown_play_and_duplicate_order_fail():
    raw,meta=fixture();raw.loc[1,'play_type']='New mystery type'
    with pytest.raises(ValueError,match='unreviewed'):college_frame(raw,meta)
    raw,meta=fixture();raw.loc[1,'game_row_number']=1
    with pytest.raises(ValueError,match='duplicate'):college_frame(raw,meta)


def test_export_does_not_change_source_frame():
    raw,meta=fixture();before=raw.copy(deep=True)
    extract_college(raw,meta)
    pd.testing.assert_frame_equal(raw,before)


def test_count_based_score_verification_catches_corruption():
    raw,meta=fixture();g=extract_college(raw,meta);g.regulation=(0,0)
    with pytest.raises(ValueError,match='score replay'):matrix(g,sorted({e.key for e in g.events}))


def test_conversion_not_ordinary_down_even_with_bad_source_down():
    assert parsed_kind('Two Point Pass',1)==('no_play',False)
    assert parsed_kind('Kickoff',1)==('no_play',False)
    assert parsed_kind('Punt',4)==('punt',True)


def test_team_view_uses_home_away_identity_and_reverses_roles():
    from read_game_rows import team_view
    r=dict(matrix_status='parsed_regulation',matrix_shape=[2,9,1],
           transition_probabilities_flat=list(range(18)),home_team_id='H',away_team_id='A')
    assert np.array_equal(team_view(r,'A'),team_view(r,'H')[::-1])
    with pytest.raises(ValueError,match='not in this game'):team_view(r,'C')


def test_loader_refuses_missing_matrix_instead_of_zero_filling():
    from read_game_rows import unpack
    with pytest.raises(ValueError,match='No usable matrix'):
        unpack(dict(matrix_status='missing_pbp',matrix_issue='awaiting source'))


def espn_fixture():
    raw,meta=fixture();rows=[]
    for r in raw.itertuples():
        seconds=int(r.TimeSecsRem)-(900 if r.period in (1,3) else 0)
        rows.append({'game_play_number':r.game_row_number,'status_type_completed':True,
            'homeTeamId':10,'awayTeamId':20,'homeFinalScore':14,'awayFinalScore':0,
            'homeScore':r.pos_team_score if r.pos_team=='H' else r.def_pos_team_score,
            'awayScore':r.pos_team_score if r.pos_team=='A' else r.def_pos_team_score,
            'period.number':r.period,'type.text':r.play_type,'start.down':r.down,
            'start.team.id':10 if r.pos_team=='H' else 20,'start.distance':10,
            'start.yardsToEndzone':50,'clock.displayValue':f'{seconds//60}:{seconds%60:02}'})
    return pd.DataFrame(rows),meta


def test_fallback_replays_correct_teams_and_preserves_decisions():
    from espn_college_adapter import extract_espn_college
    raw,meta=espn_fixture();g=extract_espn_college(raw,meta)
    assert g.regulation==(14,0) and g.final==(14,0)
    assert len(g.events)==5
    assert any(e.opponent_points==7 and not e.home_offense for e in g.events)


def test_fallback_rejects_order_reversal_and_wrong_score():
    from espn_college_adapter import extract_espn_college
    raw,meta=espn_fixture();raw.loc[3,'period.number']=2
    with pytest.raises(ValueError,match='period order'):extract_espn_college(raw,meta)
    raw,meta=espn_fixture();meta['home_final_score']=17
    with pytest.raises(ValueError,match='final mismatch'):extract_espn_college(raw,meta)
