"""Rebuild nonterminal transition tables, preserve censored boundary evidence.

No raw source endpoint is invented. Final segments have no observed successor;
retain their rewards separately, never let them become reusable stop edges.
Earlier-date imputation pools score context but never time buckets or leagues.
"""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .paths import RESET as ROOT, EXTRACTED as SOURCE
VERSION='football-only-elapsed-clock-v1'

def save(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def fill(p,counts,keys,team,league):
    """Donors [2,3,E] / [3,E]; same time band, pooled lead categories."""
    if not np.isfinite(p).all() or np.any(p<0):raise ValueError('Invalid probabilities')
    out=p.copy();marks=np.full((2,9,72),-1,dtype=np.int8)
    for s in range(72):
        ix=np.flatnonzero(keys[:,0]==s)
        for d in range(2):
            for c in range(9):
                observed=counts[d,c,ix]
                if observed.sum():
                    if not np.allclose(p[d,c,ix],observed/observed.sum(),atol=1e-7):raise ValueError('Bad observed row')
                    marks[d,c,s]=0;continue
                if np.any(p[d,c,ix]):raise ValueError('Probability without counts')
                for tag,donor in [(1,team[d,c//3,ix]),(2,league[c//3,ix])]:
                    if donor.sum():out[d,c,ix]=donor/donor.sum();marks[d,c,s]=tag;break
    return out,marks

class History:
    def __init__(self,e):
        self.e=e;self.league=np.zeros((3,e),dtype=np.int64);self.teams={}
        self.ids=defaultdict(list);self.games=[];self.latest=None
    def team(self,t):return self.teams.get(t,np.zeros((3,self.e),dtype=np.int64))
    def add(self,gid,day,home,away,counts):
        if self.latest and day<self.latest:raise ValueError('History out of order')
        for d,t in enumerate((home,away)):
            raw=counts[d].reshape(3,3,self.e).sum(axis=1)
            self.teams.setdefault(t,np.zeros_like(raw))[:]+=raw;self.league+=raw;self.ids[t].append(gid)
        self.games.append(gid);self.latest=day

def support(p,keys,starts):
    pending=[(int(not s['home_offense']),int(s['state'])) for s in starts];seen=set();missing=set()
    groups=[np.flatnonzero(keys[:,0]==s) for s in range(72)]
    while pending:
        d,s=pending.pop()
        if (d,s) in seen:continue
        seen.add((d,s));ix=groups[s];v=p[d][:,ix]
        for c in np.flatnonzero(v.sum(1)==0):missing.add((d,int(c),s))
        for j in ix[np.any(v>0,axis=0)]:pending.append((1-d if keys[j,2] else d,int(keys[j,1])))
    return dict(reachable_support_complete=not missing,unresolved_reachable_rows=len(missing),unresolved_reachable_row_keys=sorted(missing))

def rebuild():
    ROOT.mkdir(parents=True,exist_ok=True)
    for name in ['raw','data']:
        out=ROOT/name
        if out.exists() and any(out.iterdir()):raise ValueError('Refusing to overwrite dataset')
        out.mkdir(exist_ok=True)
    old=json.loads((SOURCE/'transition_set.json').read_text());oldkeys=np.asarray(old['keys'],dtype=int)
    keep=oldkeys[:,1]>=0;keys=oldkeys[keep];e=len(keys);shape=(2,9,e)
    assert (keys[:,:2]>=0).all() and (keys[:,:2]<72).all()
    cat=dict(old);cat.update(version=VERSION,keys=keys.tolist(),shape=list(shape),
        termination='Elapsed clock only; no terminal football state or sampled stop edge.',
        boundary_observations='Censored successor states retained separately, excluded from transition likelihood.',
        provenance='Subset of the preserved global descriptive catalog, not a train-only vocabulary.')
    seg=pd.read_parquet(SOURCE/'parsed_segments.parquet');normal=seg[seg.destination.ge(0)].copy();boundary=seg[seg.destination.lt(0)].copy()
    assert boundary.destination.eq(-1).all()
    boundary['destination']=pd.array([None]*len(boundary),dtype='Int64')
    boundary['successor_observed']=False;boundary['censor_reason']='no_next_decision_in_same_half'
    boundary['historical_elapsed_is_boundary_censored']=True
    for name in ['raw','data']:
        save(ROOT/name/'transition_set.json',cat)
        normal.to_parquet(ROOT/name/'parsed_segments.parquet',index=False,compression='zstd')
        boundary.to_parquet(ROOT/name/'boundary_evidence.parquet',index=False,compression='zstd')
    bgroup={k:g for k,g in boundary.groupby(['league','game_id'])}
    provenance=[];historylog=[];audit=[];summary={}
    for league in ['nfl','college']:
        src=SOURCE/f'{league}_games.parquet';schema=pq.read_schema(src);hist=History(e);pending=[];daynow=None;stats=Counter();buffers=[[],[]]
        writers=[pq.ParquetWriter(ROOT/n/f'{league}_games.parquet',schema,compression='zstd') for n in ['raw','data']]
        def commit():
            for args in pending:hist.add(*args)
            pending.clear()
        try:
            for block in pq.ParquetFile(src).iter_batches(batch_size=8):
                for row in block.to_pylist():
                    gid=row['game_id'];day=str(row['game_date'])[:10];stats['rows']+=1
                    if daynow!=day:
                        if daynow and day<daynow:raise ValueError('Input dates not sorted')
                        commit();daynow=day
                        historylog.append(dict(league=league,target_date=day,latest_donor_date=hist.latest,donor_game_ids=list(hist.games)))
                    # All rows carry the new representation version; null arrays remain null.
                    row['transition_set_version']=VERSION
                    if row['matrix_status']!='parsed_regulation':
                        # Original missing rows have no shape; preserve their nulls.
                        raw=dict(row);filledrow=dict(row);stats['null_games']+=1
                    else:
                        oldc=np.asarray(row['transition_counts_flat'],dtype=np.int64).reshape(2,9,len(oldkeys))
                        counts=oldc[:,:,keep];totals=np.zeros((2,9,72),dtype=np.int64)
                        for s in range(72):totals[:,:,s]=counts[:,:,keys[:,0]==s].sum(2)
                        den=totals[:,:,keys[:,0]]
                        p=np.divide(counts,den,out=np.zeros(shape,dtype=np.float32),where=den>0)
                        home,away=row['home_team_id'],row['away_team_id'];starts=json.loads(row['half_starts_json'])
                        rewards=counts.sum(1)@keys[:,3:5]
                        nh=int(rewards[0,0]+rewards[1,1]);na=int(rewards[0,1]+rewards[1,0])
                        b=bgroup[league,gid]
                        bh=int(np.where(b.home_offense,b.own_points,b.opponent_points).sum())
                        ba=int(np.where(b.home_offense,b.opponent_points,b.own_points).sum())
                        oh=sum(s['home_points'] for s in starts);oa=sum(s['away_points'] for s in starts)
                        if (nh+bh+oh,na+ba+oa)!=(row['home_regulation_score'],row['away_regulation_score']):raise ValueError('Lost scoring evidence')
                        raw=dict(row);raw.update(matrix_shape=list(shape),transition_probabilities_flat=p.ravel().tolist(),transition_counts_flat=counts.ravel().tolist(),source_counts_flat=totals.ravel().tolist())
                        # segment_count names the modeled segments; full evidence count lives in audit.
                        raw['segment_count']=int(counts.sum())
                        filled,marks=fill(p,counts,keys,np.stack([hist.team(home),hist.team(away)]),hist.league)
                        filledrow=dict(raw);filledrow['transition_probabilities_flat']=filled.ravel().tolist()
                        ready=support(filled,keys,starts)
                        tags={n:int((marks==t).sum()) for t,n in [(0,'observed'),(1,'earlier_team_same_time'),(2,'earlier_league_same_time'),(-1,'unresolved')]}
                        provenance.append(dict(league=league,game_id=gid,target_date=day,home_team_id=home,away_team_id=away,
                            marking_shape=[2,9,72],provenance_flat=marks.ravel().tolist(),home_donor_game_ids=list(hist.ids[home]),away_donor_game_ids=list(hist.ids[away]),latest_donor_date=hist.latest,**tags,**ready))
                        audit.append(dict(league=league,game_id=gid,original_segments=int(oldc.sum()),modeled_segments=int(counts.sum()),boundary_segments=len(b),
                            modeled_home_points=nh,modeled_away_points=na,boundary_home_points=bh,boundary_away_points=ba,
                            opening_home_points=oh,opening_away_points=oa,regulation_home_points=nh+bh+oh,regulation_away_points=na+ba+oa,
                            score_evidence_reconciled=True,boundary_successors_unresolved=len(b),**ready))
                        stats['parsed_games']+=1;stats['reachable_support_complete_games']+=ready['reachable_support_complete']
                        for n,v in tags.items():stats[n+'_rows']+=v
                        pending.append((gid,day,home,away,counts))
                    for i,v in enumerate([raw,filledrow]):buffers[i].append(v)
                    if len(buffers[0])>=8:
                        for w,buf in zip(writers,buffers):w.write_table(pa.Table.from_pylist(buf,schema=schema));buf.clear()
            for w,buf in zip(writers,buffers):
                if buf:w.write_table(pa.Table.from_pylist(buf,schema=schema))
        finally:
            for w in writers:w.close()
        summary[league]=dict(stats);print(league,dict(stats),flush=True)
    pd.DataFrame(provenance).to_parquet(ROOT/'data'/'imputation-provenance.parquet',index=False,compression='zstd')
    pd.DataFrame(audit).to_parquet(ROOT/'data'/'score-evidence-audit.parquet',index=False,compression='zstd')
    save(ROOT/'data'/'donor-history-log.json',historylog)
    save(ROOT/'data'/'summary.json',dict(version=VERSION,shape=list(shape),tables=summary,
        raw_source=str(SOURCE.resolve()),source_hashes={f:sha(SOURCE/f) for f in ['transition_set.json','parsed_segments.parquet','nfl_games.parquet','college_games.parquet']},
        imputation='Strictly earlier UTC dates; same team then same league; same time bin, pooled lead; observed rows unchanged; no recursively imputed donors.',
        boundary_limit='Last-segment rewards retained in separate censored evidence, excluded from transition fit. Matrix-only count replay no longer equals full regulation scores; evidence-combined replay must. No complete final-play generative model is claimed.',
        readiness='Design/research dataset; support completeness is not calibration or training approval.'))

if __name__=='__main__':rebuild()
