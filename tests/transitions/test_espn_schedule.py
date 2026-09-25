from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import pandas as pd
import pytest
from goalpost.transitions.espn_schedule import load, normalize, request_url
from goalpost.transitions.weekly import select_week

NOW=datetime(2026,9,25,14,tzinfo=timezone.utc)

def payload():
    return {'season':{'year':2026,'type':2},'week':{'number':3},'events':[{
        'id':'40123','date':'2026-09-27T17:00Z','season':{'year':2026,'type':2},'week':{'number':3},
        'status':{'type':{'name':'STATUS_SCHEDULED','state':'pre','completed':False}},
        'competitions':[{'competitors':[
            {'homeAway':'away','team':{'abbreviation':'LAR'},'score':'0'},
            {'homeAway':'home','team':{'abbreviation':'WSH'},'score':'0'}]}]}]}

def test_normalize_orientation_aliases_and_placeholder_scores():
    frame=normalize(payload());r=frame.iloc[0]
    assert r.game_id=='2026_03_LA_WAS' and r.espn_event_id=='40123'
    assert r.home_team=='WAS' and r.away_team=='LA'
    assert r.gametime=='13:00' and pd.isna(r.home_score) and pd.isna(r.away_score)
    assert len(select_week(frame,NOW)[0])==1

@pytest.mark.parametrize('name,state,completed',[
    ('STATUS_FINAL','post',True),('STATUS_IN_PROGRESS','in',False),
    ('STATUS_POSTPONED','pre',False),('STATUS_CANCELED','post',False)])
def test_status_is_not_inferred_from_zero_scores(name,state,completed):
    v=payload();v['events'][0]['status']['type']=dict(name=name,state=state,completed=completed)
    games,excluded,_=select_week(normalize(v),NOW,2026,3)
    assert not games and excluded[0]['reason']=='provider_status_'+name

def test_utc_to_et_date_rollover():
    v=payload();v['events'][0]['date']='2026-09-28T00:15:30Z'
    r=normalize(v).iloc[0];assert r.gameday=='2026-09-27' and r.gametime=='20:15'
    assert select_week(normalize(v),NOW)[0][0]['kickoff_utc']=='2026-09-28T00:15:30+00:00'


def test_duplicate_unknown_and_missing_timezone_fail():
    for mutation in ['duplicate','team','time']:
        v=payload()
        if mutation=='duplicate':v['events'].append(deepcopy(v['events'][0]))
        if mutation=='team':v['events'][0]['competitions'][0]['competitors'][0]['team']['abbreviation']='UNKNOWN'
        if mutation=='time':v['events'][0]['date']='2026-09-27 17:00'
        with pytest.raises(ValueError):normalize(v)


def test_cache_hit_miss_expiry_and_force(tmp_path):
    calls=[]
    def fetch(url):calls.append(url);return json.dumps(payload()).encode()
    first=load(tmp_path,now=NOW,fetcher=fetch)
    second=load(tmp_path,now=NOW+timedelta(minutes=59),fetcher=fetch)
    assert len(calls)==1 and not first[2]['cache_hit'] and second[2]['cache_hit']
    assert first[0]==second[0] and first[1]==second[1]
    load(tmp_path,now=NOW+timedelta(hours=1),fetcher=fetch);assert len(calls)==2
    load(tmp_path,now=NOW+timedelta(hours=1),refresh=True,fetcher=fetch);assert len(calls)==3


def test_failed_refresh_preserves_cache_but_never_uses_stale(tmp_path):
    _,_,meta=load(tmp_path,now=NOW,fetcher=lambda url:json.dumps(payload()).encode())
    p=Path(meta['cache_path']);before=p.read_bytes()
    def fail(url):raise OSError('offline')
    with pytest.raises(OSError):load(tmp_path,now=NOW+timedelta(hours=2),fetcher=fail)
    assert p.read_bytes()==before
    with pytest.raises(ValueError):load(tmp_path,now=NOW,refresh=True,fetcher=lambda u:b'{"bad":1}')
    assert p.read_bytes()==before


def test_corrupt_and_future_dated_cache_refetched(tmp_path):
    calls=[]
    def fetch(u):calls.append(u);return json.dumps(payload()).encode()
    _,_,meta=load(tmp_path,now=NOW,fetcher=fetch)
    p=Path(meta['cache_path']);p.write_text('{}')
    load(tmp_path,now=NOW,fetcher=fetch);assert len(calls)==2
    load(tmp_path,now=NOW-timedelta(minutes=1),fetcher=fetch);assert len(calls)==3


def test_request_key_separates_weeks_and_rejects_wrong_week_before_write(tmp_path):
    assert request_url(2026,3)!=request_url(2026,4)
    with pytest.raises(ValueError,match='different requested'):
        load(tmp_path,season=2026,week=4,now=NOW,fetcher=lambda u:json.dumps(payload()).encode())
    assert not list(tmp_path.glob('*.json'))


def test_postseason_provider_round_to_nfl_week():
    v=payload();v['events'][0]['season']['type']=3;v['events'][0]['week']['number']=5
    assert normalize(v).iloc[0].game_type=='SB' and normalize(v).iloc[0].week==22
    assert 'seasontype=3' in request_url(2026,22,'SB') and 'week=5' in request_url(2026,22,'SB')


def test_cli_automatic_cache_without_csv(tmp_path,monkeypatch):
    import goalpost.transitions.weekly as w
    import goalpost.transitions.espn_schedule as e
    calls=[]
    def fetch(u):calls.append(u);return json.dumps(payload()).encode()
    monkeypatch.setattr(e,'fetch',fetch)
    # No upcoming games at the future as-of: confirms cache uses wall clock,
    # CLI normalization and saved raw evidence without requiring matrix data.
    for name in ['first','second']:
        assert w.main(['--output',str(tmp_path/name),'--schedule-cache',str(tmp_path/'cache'),
                       '--as-of','2030-01-01T00:00:00Z'])==2
    assert len(calls)==1
    report=json.loads((tmp_path/'second/report.json').read_text())
    assert report['schedule']['cache_hit']
    assert (tmp_path/'second/schedule-espn.json').exists() and (tmp_path/'second/schedule.csv').exists()
