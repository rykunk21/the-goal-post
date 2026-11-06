"""One row per game; two identical schemas; explicit rejected/missing matrices."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .extraction import extract_game, replay, N_STATES, Event, ExtractedGame
from .college_adapter import extract_college
from .espn_college_adapter import extract_espn_college
from .raw_espn_college import extract_raw_espn

CFB_COLUMNS=['game_id','game_row_number','id_play','year','week','home','away',
 'home_team_id','away_team_id','pos_team','def_pos_team','pos_team_score','def_pos_team_score',
 'period','TimeSecsRem','down','distance','yards_to_goal','play_type']
CUTOFF='2026-09-23T23:59:59Z'


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def schedules(root):
    rows=[]
    d=pd.read_csv(root/'nfl-schedule.csv')
    d=d[d.season.between(2023,2026) & d.home_score.notna() & d.away_score.notna() &
        (d.game_type!='PRE') & (pd.to_datetime(d.gameday,utc=True)<=pd.Timestamp(CUTOFF))]
    for r in d.itertuples():
        rows.append(dict(league='nfl',game_id=str(r.game_id),season=int(r.season),week=int(r.week),
            season_type=r.game_type,game_date=str(r.gameday),date_precision='calendar_date',
            home_team=str(r.home_team),away_team=str(r.away_team),
            home_team_id=str(r.home_team),away_team_id=str(r.away_team),
            home_division='nfl',away_division='nfl',home_final_score=int(r.home_score),
            away_final_score=int(r.away_score),schedule_source='nflverse/nfldata',
            matrix_status='missing_pbp',matrix_issue='no parsed source game'))
    for year in range(2023,2027):
        d=pd.read_parquet(root/f'cfb_schedules-{year}.parquet')
        d=d[(d.completed==True) & d.home_points.notna() & d.away_points.notna() &
            (d.home_division.isin(['fbs','fcs']) | d.away_division.isin(['fbs','fcs'])) &
            (pd.to_datetime(d.start_date,utc=True)<=pd.Timestamp(CUTOFF))]
        for r in d.itertuples():
            rows.append(dict(league='college',game_id=str(r.game_id),season=int(r.season),week=int(r.week),
                season_type=str(r.season_type),game_date=str(r.start_date),date_precision='scheduled_utc',
                home_team=str(r.home_team),away_team=str(r.away_team),
                home_team_id=str(r.home_id),away_team_id=str(r.away_id),
                home_division=str(r.home_division),away_division=str(r.away_division),
                home_final_score=int(r.home_points),away_final_score=int(r.away_points),
                schedule_source='sportsdataverse/cfb_schedules',matrix_status='missing_pbp',matrix_issue='no parsed source game'))
    identities=[(r['league'],r['game_id']) for r in rows]
    if len(set(identities))!=len(identities):raise ValueError('Duplicate schedule identity')
    return sorted(rows,key=lambda r:(r['league'],r['game_date'],r['game_id']))


def matrix(game,keys):
    index={k:i for i,k in enumerate(keys)}
    counts=np.zeros((2,9,len(keys)),dtype=np.int32)
    totals=np.zeros((2,9,N_STATES),dtype=np.int32)
    for e in game.events:
        direction=0 if e.home_offense else 1
        counts[direction,e.ctx,index[e.key]]+=1
        totals[direction,e.ctx,e.source]+=1
    sources=np.array([k[0] for k in keys])
    denominators=totals[...,sources]
    probabilities=np.divide(counts,denominators,out=np.zeros_like(counts,dtype=np.float32),where=denominators>0)
    points=np.array([[k[3],k[4]] for k in keys])
    home_rewards=counts[0].sum(0)@points
    away_rewards=counts[1].sum(0)@points
    rebuilt=(int(home_rewards[0]+away_rewards[1]+sum(s['home_points'] for s in game.starts)),
             int(home_rewards[1]+away_rewards[0]+sum(s['away_points'] for s in game.starts)))
    if rebuilt!=game.regulation:raise ValueError('Matrix count/reward score replay mismatch')
    return probabilities.ravel(),counts.ravel(),totals.ravel()


def export(root,nfl2023,out,reuse=None,raw_summaries=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use an empty output directory')
    rows=schedules(root);lookup={(r['league'],r['game_id']):r for r in rows}
    parsed={}; source_omissions=[]; sources=[]
    for row in rows:row['pbp_source']=None
    if reuse:
        previous=json.loads((reuse/'coverage.json').read_text())
        sources=previous['sources'];source_omissions=previous['source_games_not_in_inventory']
        for source in sources:
            p=nfl2023 if source['league']=='nfl' and source['season']==2023 else root/source['file']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=source['sha256']:
                raise ValueError('Changed source; cannot reuse parsed evidence')
        seg=pd.read_parquet(reuse/'parsed_segments.parquet')
        groups={k:g for k,g in seg.groupby(['league','game_id'])}
        seen=set()
        for league in ['nfl','college']:
            cols=[c for c in pq.read_schema(reuse/f'{league}_games.parquet').names
                  if c not in ['transition_probabilities_flat','transition_counts_flat','source_counts_flat']]
            for old in pd.read_parquet(reuse/f'{league}_games.parquet',columns=cols).to_dict('records'):
                if old.get('pbp_source')=='espn-summary-raw':
                    raise ValueError('Reuse the frozen v2 predecessor; raw-summary games must be re-extracted with their source maps')
                key=(league,old['game_id']);seen.add(key);meta=lookup[key]
                for field in ['home_final_score','away_final_score','home_team_id','away_team_id','game_date']:
                    if old[field]!=meta[field]:raise ValueError('Changed schedule; cannot reuse parsed evidence')
                meta.update(matrix_status=old['matrix_status'],matrix_issue=None if pd.isna(old['matrix_issue']) else old['matrix_issue'])
                if old['matrix_status']=='parsed_regulation':
                    events=[Event(**{k:r[k] for k in Event.__dataclass_fields__}) for r in groups[key].to_dict('records')]
                    reg=(int(old['home_regulation_score']),int(old['away_regulation_score']))
                    final=(old['home_final_score'],old['away_final_score'])
                    parsed[key]=ExtractedGame(old['game_id'],old['week'],old['game_date'],old['home_team'],old['away_team'],
                        events,json.loads(old['half_starts_json']),reg,final,reg!=final)
                    meta['pbp_source']=old['pbp_source']
        if seen!=set(lookup):raise ValueError('Changed inventory; cannot reuse parsed evidence')
    for league in ([] if reuse else ['nfl','college']):
        for year in range(2023,2027):
            p=nfl2023 if league=='nfl' and year==2023 else root/f'{"nfl" if league=="nfl" else "cfb"}-{year}.parquet'
            sources.append(dict(file=p.name,league=league,season=year,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            frame=pd.read_parquet(p,columns=CFB_COLUMNS if league=='college' else None)
            for gid,g in frame.groupby('game_id',sort=True):
                key=(league,str(gid))
                if key not in lookup:
                    source_omissions.append(dict(league=league,game_id=str(gid),season=year,reason='not in completed in-scope schedule inventory'))
                    continue
                meta=lookup[key]
                try:
                    game=extract_game(g) if league=='nfl' else extract_college(g,meta)
                    if game.final!=(meta['home_final_score'],meta['away_final_score']):
                        raise ValueError('Schedule/final mismatch')
                    if league=='nfl' and (game.home,game.away)!=(meta['home_team'],meta['away_team']):
                        raise ValueError('Schedule/team orientation mismatch')
                    if str(game.date)[:10]!=str(meta['game_date'])[:10]:
                        raise ValueError('Schedule/game date mismatch')
                    if replay(game)!=game.regulation:raise ValueError('Event score replay mismatch')
                    parsed[key]=game
                    meta.update(matrix_status='parsed_regulation',matrix_issue=None,
                        pbp_source='nflverse' if league=='nfl' else 'cfbfastR')
                except (ValueError,TypeError,KeyError) as e:
                    meta.update(matrix_status='rejected',matrix_issue=str(e))
            print(league,year,dict(Counter(r['matrix_status'] for r in rows if r['league']==league and r['season']==year)),flush=True)
    # Explicit whole-game fallback; never splice plays or overwrite a valid
    # primary parse. This source currently covers only part of college scope.
    fallback_path=root/'espn_cfb_pbp-2026.parquet'
    fallback_audit=[]
    if fallback_path.exists():
        sources.append(dict(file=fallback_path.name,league='college',season=2026,
            sha256=hashlib.sha256(fallback_path.read_bytes()).hexdigest(),role='whole-game fallback'))
        needed=['game_id','game_play_number','status_type_completed','homeTeamId','awayTeamId','homeFinalScore','awayFinalScore',
                'homeScore','awayScore','period.number','type.text','start.down','start.team.id','start.distance',
                'start.yardsToEndzone','clock.displayValue']
        for gid,g in pd.read_parquet(fallback_path,columns=needed).groupby('game_id'):
            key=('college',str(gid))
            if key not in lookup or key in parsed:continue
            meta=lookup[key];audit=dict(game_id=str(gid),primary_issue=meta['matrix_issue'])
            try:
                game=extract_espn_college(g,meta)
                if replay(game)!=game.regulation:raise ValueError('Fallback event replay mismatch')
                parsed[key]=game;meta.update(matrix_status='parsed_regulation',matrix_issue=None,pbp_source='espn-derived')
                audit['result']='accepted'
            except (ValueError,TypeError,KeyError) as e:
                audit.update(result='rejected',fallback_issue=str(e))
            fallback_audit.append(audit)
        print('college 2026 fallback',dict(Counter(x['result'] for x in fallback_audit)),flush=True)
    raw_audit=[]; source_rows=[]
    if raw_summaries:
        for meta in rows:
            key=(meta['league'],meta['game_id'])
            if meta['league']!='college' or meta['season']!=2026 or key in parsed:continue
            p=raw_summaries/(meta['game_id']+'.json')
            if not p.exists():continue
            evidence=p.read_bytes()
            audit=dict(game_id=meta['game_id'],primary_issue=meta['matrix_issue'],
                       file=str(p.resolve()),sha256=hashlib.sha256(evidence).hexdigest())
            try:
                game,mapping=extract_raw_espn(json.loads(evidence),meta)
                if replay(game)!=game.regulation:raise ValueError('Raw event replay mismatch')
                parsed[key]=game
                meta.update(matrix_status='parsed_regulation',matrix_issue=None,pbp_source='espn-summary-raw')
                audit.update(result='accepted',segments=len(game.events),
                    zero_elapsed_segments=sum(e.elapsed==0 for e in game.events))
                source_rows.extend(dict(game_id=meta['game_id'],**row) for row in mapping)
            except (ValueError,TypeError,KeyError,AttributeError) as e:
                audit.update(result='rejected',fallback_issue=str(e))
            raw_audit.append(audit)
        print('college 2026 raw fallback',dict(Counter(x['result'] for x in raw_audit)),flush=True)
    # Descriptive design catalog, NOT a vocabulary fitted for a held-out model.
    # Every parsed event is retained; no observed transition is projected away.
    keys=sorted({e.key for game in parsed.values() for e in game.events})
    version=hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest()
    schema=pa.schema([(k,pa.string()) for k in ['league','game_id','game_date','date_precision','season_type',
        'home_team','away_team','home_team_id','away_team_id','home_division','away_division',
        'schedule_source','pbp_source','matrix_status','matrix_issue','transition_set_version','transition_scope','clock_semantics']]+
        [(k,pa.int32()) for k in ['season','week','home_final_score','away_final_score',
            'home_regulation_score','away_regulation_score','segment_count']]+
        [('matrix_shape',pa.list_(pa.int32())),('transition_probabilities_flat',pa.list_(pa.float32())),
         ('transition_counts_flat',pa.list_(pa.int32())),('source_counts_flat',pa.list_(pa.int32())),
         ('half_starts_json',pa.string())])
    segment_records=[]
    for league in ['nfl','college']:
        with pq.ParquetWriter(out/f'{league}_games.parquet',schema,compression='zstd') as writer:
            batch=[]
            for row in [r for r in rows if r['league']==league]:
                row.update(transition_set_version=version,transition_scope='regulation',matrix_shape=[2,9,len(keys)],
                    clock_semantics='nflverse predecision half-clock' if league=='nfl' else 'college provider clock; may be end-of-play')
                game=parsed.get((league,row['game_id']))
                if game:
                    p,c,t=matrix(game,keys)
                    row.update(transition_probabilities_flat=p.tolist(),transition_counts_flat=c.tolist(),
                        source_counts_flat=t.tolist(),half_starts_json=json.dumps(game.starts,default=lambda x:int(x)),
                        home_regulation_score=int(game.regulation[0]),away_regulation_score=int(game.regulation[1]),segment_count=len(game.events))
                    segment_records.extend(dict(league=league,**asdict(e)) for e in game.events)
                else:
                    row.update(transition_probabilities_flat=None,transition_counts_flat=None,source_counts_flat=None,
                        half_starts_json=None,home_regulation_score=None,away_regulation_score=None,segment_count=None)
                batch.append({k:row[k] for k in schema.names})
                for name in ['transition_probabilities_flat','transition_counts_flat','source_counts_flat']:
                    row.pop(name,None)
                if len(batch)==8:
                    writer.write_table(pa.Table.from_pylist(batch,schema=schema));batch=[]
            if batch:writer.write_table(pa.Table.from_pylist(batch,schema=schema))
    pd.DataFrame(segment_records).to_parquet(out/'parsed_segments.parquet',index=False,compression='zstd')
    write_json(out/'transition_set.json',dict(version=version,keys=keys,
        key_fields=['source','destination','switch','own_points','opponent_points'],shape=[2,9,len(keys)],
        flatten_order='C: home-offense then away-offense; within each, 9 contexts then ordered edge keys',
        source_counts_shape=[2,9,72],source_mask='source_counts > 0',
        catalog_scope='Union of parsed observations, including later seasons. Descriptive transition-design export, not a training-only vocabulary.',
        unknown_games='Null matrix on rejected/missing rows; not a zero-probability game'))
    coverage=[]
    for league in ['nfl','college']:
        for year in range(2023,2027):
            selected=[r for r in rows if r['league']==league and r['season']==year]
            coverage.append(dict(league=league,season=year,scheduled_completed_games=len(selected),
                matrix_statuses=dict(Counter(r['matrix_status'] for r in selected)),
                latest_scheduled_date=max((r['game_date'] for r in selected),default=None),
                latest_parsed_date=max((r['game_date'] for r in selected if r['matrix_status']=='parsed_regulation'),default=None),
                standalone_fcs_games=sum(r['home_division']=='fcs' and r['away_division']=='fcs' for r in selected),
                standalone_fcs_parsed=sum(r['home_division']=='fcs' and r['away_division']=='fcs' and r['matrix_status']=='parsed_regulation' for r in selected)))
    write_json(out/'coverage.json',dict(cutoff=CUTOFF,rows=coverage,sources=sources,source_games_not_in_inventory=source_omissions,
        limitation='Inventory is published schedule coverage, not independently proven exhaustive; missing/rejected rows are not trainable. No scheduled updater or full training run.'))
    write_json(out/'exceptions.json',[{k:r[k] for k in ['league','game_id','season','home_team','away_team','game_date','matrix_status','matrix_issue']}
                                     for r in rows if r['matrix_status']!='parsed_regulation'])
    write_json(out/'fallback_audit.json',fallback_audit)
    write_json(out/'raw_fallback_audit.json',raw_audit)
    pd.DataFrame(source_rows).to_parquet(out/'raw_source_row_map.parquet',index=False,compression='zstd')
    print(json.dumps(coverage,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--nfl2023',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-parsed',type=Path)
    p.add_argument('--raw-summaries',type=Path)
    a=p.parse_args();export(a.sources,a.nfl2023,a.output,a.reuse_parsed,a.raw_summaries)
