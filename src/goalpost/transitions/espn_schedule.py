"""ESPN scoreboard normalization and atomic, expiring local response cache."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os
import tempfile
import urllib.parse
import urllib.request

import pandas as pd

URL = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'
VERSION = 'espn-schedule-v1'
TEAMS = set('ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS'.split())
ALIASES = {'LAR': 'LA', 'WSH': 'WAS'}
POST = {1: 'WC', 2: 'DIV', 3: 'CON', 5: 'SB'}
COLUMNS = ['game_id','season','week','game_type','gameday','gametime','home_team',
           'away_team','home_score','away_score','espn_event_id','source_status','kickoff_utc']


def request_url(season=None, week=None, game_type='REG'):
    if (season is None) != (week is None):
        raise ValueError('Supply both season and week')
    if season is None:
        return URL
    if game_type == 'REG':
        provider_week = week; season_type = 2
    else:
        expected = (18 if season >= 2021 else 17) + ['WC','DIV','CON','SB'].index(game_type) + 1
        if week != expected:
            raise ValueError('Postseason --week must use the NFL season week, not ESPN round numbering')
        provider_week = {'WC':1,'DIV':2,'CON':3,'SB':5}[game_type]; season_type = 3
    return URL + '?' + urllib.parse.urlencode(dict(dates=season,seasontype=season_type,week=provider_week,limit=100))


def normalize(payload):
    if not isinstance(payload,dict) or not isinstance(payload.get('events'),list):
        raise ValueError('ESPN response must contain an events list')
    rows=[]; ids=set()
    for event in payload['events']:
        eid=str(event['id'])
        if not eid.isdigit() or eid in ids:
            raise ValueError('Invalid or duplicate ESPN event ID')
        ids.add(eid)
        competitions=event['competitions']
        if len(competitions)!=1:
            raise ValueError('Ambiguous ESPN competition')
        competitors=competitions[0]['competitors']
        if len(competitors)!=2 or {x['homeAway'] for x in competitors}!={'home','away'}:
            raise ValueError('Expected exact home and away competitors')
        by_side={x['homeAway']:x for x in competitors}
        teams={side:ALIASES.get(x['team']['abbreviation'],x['team']['abbreviation']) for side,x in by_side.items()}
        if any(t not in TEAMS for t in teams.values()) or teams['home']==teams['away']:
            raise ValueError('Unrecognized NFL team identity')
        season=int(event.get('season',payload.get('season',{}))['year'])
        st=int(event.get('season',payload.get('season',{}))['type'])
        week=int(event.get('week',payload.get('week',{}))['number'])
        if st==2:
            kind='REG'
        elif st==3 and week in POST:
            kind=POST[week]; week=(18 if season>=2021 else 17)+['WC','DIV','CON','SB'].index(kind)+1
        elif st==1:
            kind='PRE'
        else:
            # Pro Bowl and unknown season types are not silently NFL matchups.
            raise ValueError('Unsupported ESPN season/round')
        kickoff=pd.Timestamp(event['date'])
        if pd.isna(kickoff) or kickoff.tzinfo is None:
            raise ValueError('Missing timezone-aware kickoff')
        kickoff=kickoff.tz_convert('UTC');local=kickoff.tz_convert('America/New_York')
        status=event['status']['type'];name=status['name']
        # Only explicitly scheduled events are candidates, including 0-0 placeholders.
        scheduled=name=='STATUS_SCHEDULED' and status.get('state')=='pre' and status.get('completed') is False
        scores={side:None if scheduled else pd.to_numeric(x.get('score'),errors='coerce') for side,x in by_side.items()}
        rows.append(dict(game_id=f"{season}_{week:02d}_{teams['away']}_{teams['home']}",season=season,week=week,
            game_type=kind,gameday=local.strftime('%Y-%m-%d'),gametime=local.strftime('%H:%M'),
            home_team=teams['home'],away_team=teams['away'],home_score=scores['home'],away_score=scores['away'],
            espn_event_id=eid,source_status='scheduled' if scheduled else name,kickoff_utc=kickoff.isoformat()))
    frame=pd.DataFrame(rows,columns=COLUMNS)
    if frame.game_id.duplicated().any():
        raise ValueError('Multiple ESPN events map to the same NFL game ID')
    return frame


def fetch(url):
    with urllib.request.urlopen(url,timeout=30) as response:
        raw=response.read(5*1024*1024+1)
    if len(raw)>5*1024*1024:
        raise ValueError('ESPN response exceeds size limit')
    return raw


def atomic_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.espn-')
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def load(cache_dir, season=None, week=None, game_type='REG', max_age_seconds=3600,
         refresh=False, now=None, fetcher=None):
    """Cache TTL uses wall-clock retrieval time, never a retrospective --as-of."""
    if not math.isfinite(max_age_seconds) or max_age_seconds<0:
        raise ValueError('Cache age must be finite and nonnegative')
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None:raise ValueError('Cache clock requires timezone')
    url=request_url(season,week,game_type)
    path=Path(cache_dir)/(hashlib.sha256(url.encode()).hexdigest()+'.json')
    entry=None;reason='forced_refresh' if refresh else 'missing'
    if path.exists() and not refresh:
        try:
            candidate=json.loads(path.read_text());raw=candidate['raw_json'].encode()
            at=datetime.fromisoformat(candidate['retrieved_at_utc']);age=(now-at).total_seconds()
            if candidate['version']!=VERSION or candidate['url']!=url or hashlib.sha256(raw).hexdigest()!=candidate['raw_sha256']:
                raise ValueError('Cache identity/hash mismatch')
            cached_frame=normalize(json.loads(raw))
            if season is not None and len(cached_frame) and not (cached_frame.season.eq(season)&cached_frame.week.eq(week)&cached_frame.game_type.eq(game_type)).all():
                raise ValueError('Cached season/week/type mismatch')
            if 0<=age<max_age_seconds:
                entry=candidate
            else:reason='expired_or_future_dated'
        except (ValueError,KeyError,TypeError):reason='invalid_cache'
    if entry is None:
        raw=(fetcher or fetch)(url)
        payload=json.loads(raw);frame=normalize(payload)  # Validate before replacing prior evidence.
        if season is not None and len(frame) and not (frame.season.eq(season)&frame.week.eq(week)&frame.game_type.eq(game_type)).all():
            raise ValueError('ESPN returned a different requested season/week/type')
        text=raw.decode('utf-8')
        entry=dict(version=VERSION,url=url,retrieved_at_utc=now.isoformat(),raw_json=text,
                   raw_sha256=hashlib.sha256(raw).hexdigest())
        atomic_write(path,entry);hit=False
    else:
        raw=entry['raw_json'].encode();payload=json.loads(raw);frame=normalize(payload);hit=True
    if season is not None and len(frame):
        if not (frame.season.eq(season)&frame.week.eq(week)&frame.game_type.eq(game_type)).all():
            raise ValueError('ESPN returned a different requested season/week/type')
    csv=frame.to_csv(index=False).encode()
    metadata=dict(source=url,provider='espn',cache_hit=hit,cache_miss_reason=None if hit else reason,
        cache_path=str(path.resolve()),retrieved_at_utc=entry['retrieved_at_utc'],
        cache_age_seconds=(now-datetime.fromisoformat(entry['retrieved_at_utc'])).total_seconds(),
        max_cache_age_seconds=max_age_seconds,raw_sha256=entry['raw_sha256'],
        sha256=hashlib.sha256(csv).hexdigest(),freshness='valid_cache' if hit else 'live_download')
    return csv,raw,metadata
