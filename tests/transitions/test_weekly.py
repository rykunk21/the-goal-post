import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from goalpost.transitions.weekly import Dataset, select_week, game_seed, run_slate, main

AT='2026-09-25T14:00:00Z'
def schedule():
    return pd.DataFrame([
        dict(game_id='thursday',season=2026,week=3,game_type='REG',gameday='2026-09-24',gametime='20:15',home_team='GB',away_team='ATL',home_score=20,away_score=10),
        dict(game_id='sunday',season=2026,week=3,game_type='REG',gameday='2026-09-27',gametime='13:00',home_team='H',away_team='A',home_score=np.nan,away_score=np.nan),
        dict(game_id='monday',season=2026,week=3,game_type='REG',gameday='2026-09-28',gametime='20:15',home_team='C',away_team='D',home_score=np.nan,away_score=np.nan),
        dict(game_id='next',season=2026,week=4,game_type='REG',gameday='2026-10-01',gametime='20:15',home_team='E',away_team='F',home_score=np.nan,away_score=np.nan)])

def data(tmp):
    root=tmp/'data';root.mkdir()
    keys=[[16,16,0,7,0],[16,16,0,3,0]]
    (root/'transition_set.json').write_text(json.dumps({'keys':keys}))
    p=np.zeros((2,9,2));p[0,:,0]=1;p[1,:,1]=1
    rows=[]
    for gid,date,home,away in [('earlier','2026-09-20','H','A'),('future','2026-09-28','H','A'),('sameday','2026-09-25','H','A')]:
        rows.append(dict(game_id=gid,game_date=date,home_team_id=home,away_team_id=away,season=2026,
            matrix_status='parsed_regulation',matrix_shape=[2,9,2],transition_probabilities_flat=p.flatten().tolist(),home_final_score=999999,away_final_score=999999))
    pd.DataFrame(rows).to_parquet(root/'nfl_games.parquet')
    seg=[]
    for gid in ['earlier','future','sameday']:
        for j in range(2):
            for lead in [-1,0,1]:
                for remain in [1800,100,20]:
                    seg.append(dict(league='nfl',game_id=gid,source=16,destination=16,switch=False,own_points=keys[j][3],opponent_points=0,remaining=remain,lead=lead,elapsed=1800. if gid=='earlier' else 0.))
    pd.DataFrame(seg).to_parquet(root/'parsed_segments.parquet')
    return root

def game():return select_week(schedule(),AT)[0][0]

def test_week_selection_orientation_and_et():
    games,excluded,selection=select_week(schedule(),AT)
    assert [g['game_id'] for g in games]==['sunday','monday']
    assert games[0]['home']=='H' and games[0]['away']=='A'
    assert games[0]['kickoff_utc']=='2026-09-27T17:00:00+00:00'
    assert selection['week']==3 and excluded[0]['game_id']=='thursday'

def test_no_future_week_is_not_success():
    assert select_week(schedule(),'2027-03-01T00:00:00Z')[2]['status']=='no_upcoming_week_within_seven_days'

def test_invalid_duplicate_and_unsafe_identifiers():
    with pytest.raises(ValueError,match='duplicate'):
        select_week(pd.concat([schedule(),schedule()]),AT)
    f=schedule();f.loc[1,'game_id']='../bad'
    assert any(x['reason']=='unsafe_game_identifier' for x in select_week(f,AT,2026,3)[1])

def test_missing_kickoff_and_started_excluded():
    f=schedule();f.loc[1,'gametime']=np.nan
    assert any(x['reason']=='missing_or_invalid_kickoff' for x in select_week(f,AT,2026,3)[1])
    f=schedule();f.loc[0,['home_score','away_score']]=np.nan
    assert any(x['reason']=='kickoff_at_or_before_as_of' for x in select_week(f,AT,2026,3)[1])

def test_postseason_keeps_season_identity():
    f=schedule().iloc[[1]].copy();f.loc[:,'season']=2026;f.loc[:,'game_type']='WC';f.loc[:,'week']=19;f.loc[:,'gameday']='2027-01-10'
    g,_,s=select_week(f,'2027-01-08T00:00:00Z')
    assert s['season']==2026 and g[0]['game_type']=='WC'

def test_timezone_and_partial_selection_rejected():
    with pytest.raises(ValueError):select_week(schedule(),'2026-09-25')
    with pytest.raises(ValueError):select_week(schedule(),AT,2026)

def test_blend_and_timing_exclude_same_day_future_and_target(tmp_path):
    d=Dataset(data(tmp_path));k,p,t,prov=d.prepare(game(),AT)
    assert [x['game_id'] for x in prov['teams']['H']['games']]==['earlier']
    assert prov['timing_game_ids']==['earlier'] and all(v==[1800.] or v==[1800.]*3 for v in t.values())
    assert np.all(p[0,:,0]==1) and np.all(p[1,:,1]==1)
    g=game();g['home'],g['away']='A','H'
    _,swapped,_,_=d.prepare(g,AT)
    np.testing.assert_array_equal(swapped,p[::-1])
    g['game_id']='earlier'
    with pytest.raises(ValueError,match='No earlier'):d.prepare(g,AT)

def test_missing_team_is_blocked_not_invented(tmp_path):
    d=Dataset(data(tmp_path));g=game();g['away']='MISSING'
    with pytest.raises(ValueError,match='MISSING'):d.prepare(g,AT)

def test_catalog_validation(tmp_path):
    root=data(tmp_path);(root/'transition_set.json').write_text(json.dumps({'keys':[[16,-1,0,7,0]]}))
    with pytest.raises(ValueError,match='Football-only'):Dataset(root)

def test_game_seed_is_order_independent():
    assert game_seed(1,'a')==game_seed(1,'a')
    assert game_seed(1,'a')!=game_seed(1,'b') and game_seed(1,'a')!=game_seed(2,'a')

def test_slate_outputs_and_partial_failure(tmp_path):
    d=Dataset(data(tmp_path));out=tmp_path/'out';out.mkdir()
    games=select_week(schedule(),AT)[0]
    results=run_slate(d,games,out,AT,n=12)
    assert [r['status'] for r in results]==['simulated','blocked']
    scores=np.load(out/'sunday/scores.npy');np.testing.assert_array_equal(scores,np.tile([7,3],(12,1)))
    pmf=pd.read_csv(out/'sunday/home_minus_away-pmf.csv');assert pmf.points.tolist()==[4]
    assert (out/'sunday/distributions.png').stat().st_size>1000
    assert (out/'monday/result.json').exists()

def test_cli_uses_frozen_schedule_and_preserves_output(tmp_path):
    root=data(tmp_path);f=tmp_path/'schedule.csv';schedule().iloc[[1]].to_csv(f,index=False);out=tmp_path/'run'
    args=['--schedule',str(f),'--data',str(root),'--output',str(out),'--as-of',AT,'--simulations','3']
    assert main(args)==0
    r=json.loads((out/'report.json').read_text());assert r['status']=='complete'
    assert r['dataset_hashes'] and r['schedule']['freshness']=='local_snapshot_age_unknown'
    with pytest.raises(FileExistsError):main(args)


def test_empty_slate_records_no_data_and_exit_two(tmp_path):
    f=tmp_path/'schedule.csv';schedule().to_csv(f,index=False);out=tmp_path/'empty'
    assert main(['--schedule',str(f),'--output',str(out),'--as-of','2027-03-01T00:00:00Z'])==2
    assert json.loads((out/'report.json').read_text())['status']=='no_scheduled_matchups'


def test_failed_fetch_records_error_without_substituting_schedule(tmp_path, monkeypatch):
    import goalpost.transitions.weekly as w
    def fail(*args,**kwargs):raise OSError('offline')
    monkeypatch.setattr(w.urllib.request,'urlopen',fail)
    out=tmp_path/'failed'
    with pytest.raises(OSError,match='offline'):main(['--output',str(out),'--as-of',AT])
    assert json.loads((out/'report.json').read_text())['status']=='failed'
    assert not (out/'schedule.csv').exists()


def test_zero_completed_run_has_no_plot(tmp_path, monkeypatch):
    import goalpost.transitions.weekly as w
    d=Dataset(data(tmp_path));out=tmp_path/'zero';out.mkdir()
    def fail_rows(*args,**kwargs):
        return np.full((3,2),-1),dict(attempted=3,completed=0,failed=3,grace_completed_games=0),[],[]
    monkeypatch.setattr(w,'run',fail_rows)
    result=run_slate(d,[game()],out,AT,n=3)[0]
    assert result['status']=='no_completed_simulations' and result['n']==0
    assert not (out/'sunday/distributions.png').exists()


def test_stable_replay_uses_no_outcome_labels(tmp_path):
    d=Dataset(data(tmp_path));out1=tmp_path/'one';out2=tmp_path/'two';out1.mkdir();out2.mkdir()
    run_slate(d,[game()],out1,AT,n=5)
    table=d.data/'nfl_games.parquet';rows=pd.read_parquet(table)
    rows['home_final_score']=-999;rows['away_final_score']=1000000;rows.to_parquet(table)
    run_slate(Dataset(d.data),[game()],out2,AT,n=5)
    np.testing.assert_array_equal(np.load(out1/'sunday/scores.npy'),np.load(out2/'sunday/scores.npy'))
