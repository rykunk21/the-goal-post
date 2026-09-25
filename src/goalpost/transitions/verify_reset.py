"""Independent reconstruction from preserved observations and chronological donors."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .paths import RESET as ROOT, EXTRACTED as SOURCE

def verify():
    catalog=json.loads((ROOT/'data'/'transition_set.json').read_text());keys=np.array(catalog['keys']);e=len(keys);shape=(2,9,e)
    assert np.all((keys[:,:2]>=0)&(keys[:,:2]<72))
    assert json.loads((ROOT/'raw'/'transition_set.json').read_text())==catalog
    srcseg=pd.read_parquet(SOURCE/'parsed_segments.parquet');normal=pd.read_parquet(ROOT/'data'/'parsed_segments.parquet')
    boundary=pd.read_parquet(ROOT/'data'/'boundary_evidence.parquet')
    pd.testing.assert_frame_equal(normal.reset_index(drop=True),srcseg[srcseg.destination.ge(0)].reset_index(drop=True))
    ob=srcseg[srcseg.destination.eq(-1)].reset_index(drop=True)
    pd.testing.assert_frame_equal(boundary[ob.columns.drop('destination')].reset_index(drop=True),ob.drop(columns='destination'))
    assert boundary.destination.isna().all() and not boundary.successor_observed.any()
    assert len(normal)+len(boundary)==len(srcseg)
    for name in ['parsed_segments.parquet','boundary_evidence.parquet']:
        assert (ROOT/'data'/name).read_bytes()==(ROOT/'raw'/name).read_bytes()
    groups={k:g for k,g in normal.groupby(['league','game_id'])};bgroups={k:g for k,g in boundary.groupby(['league','game_id'])}
    index={tuple(k):j for j,k in enumerate(keys)};source=keys[:,0];times=np.repeat(np.arange(3),3)
    marks=pd.read_parquet(ROOT/'data'/'imputation-provenance.parquet').set_index(['league','game_id'])
    audit=pd.read_parquet(ROOT/'data'/'score-evidence-audit.parquet').set_index(['league','game_id'])
    changed={'transition_set_version','matrix_shape','transition_probabilities_flat','transition_counts_flat','source_counts_flat','segment_count'}
    checked=0;nulls=0;all_meta={};rows_checked=0
    def batches(path):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4):yield from batch.to_pylist()
    for league in ['nfl','college']:
        assert pq.read_schema(ROOT/'data'/f'{league}_games.parquet')==pq.read_schema(SOURCE/f'{league}_games.parquet')
        totals=np.zeros((3,e),dtype=np.int64);teams={};ids=defaultdict(list);gameids=[];daynow=None;pending=[];latest=None
        meta=pd.read_parquet(SOURCE/f'{league}_games.parquet',columns=['game_id','game_date','matrix_status','home_team_id','away_team_id']);meta['day']=meta.game_date.str[:10];all_meta[league]=meta
        count_rows=0
        for old,raw,new in zip(batches(SOURCE/f'{league}_games.parquet'),batches(ROOT/'raw'/f'{league}_games.parquet'),batches(ROOT/'data'/f'{league}_games.parquet'),strict=True):
            rows_checked+=1;count_rows+=1;gid=old['game_id'];day=old['game_date'][:10]
            if daynow!=day:
                for pg,pday,ht,at,c in pending:
                    for d,t in enumerate((ht,at)):
                        v=c[d].reshape(3,3,e).sum(1);teams.setdefault(t,np.zeros_like(v))[:]+=v;totals+=v;ids[t].append(pg)
                    gameids.append(pg);latest=pday
                pending=[];daynow=day
            for col in old:
                if col not in changed:assert raw[col]==old[col] and new[col]==old[col],(league,gid,col)
                if col!='transition_probabilities_flat':assert raw[col]==new[col],(league,gid,col)
            if old['matrix_status']!='parsed_regulation':
                assert raw['transition_probabilities_flat'] is None and new['transition_probabilities_flat'] is None;nulls+=1;continue
            g=groups[league,gid];c=np.zeros(shape,dtype=np.int64)
            js=np.array([index[(r.source,r.destination,int(r.switch),r.own_points,r.opponent_points)] for r in g.itertuples()])
            contexts=np.where(g.remaining>120,0,np.where(g.remaining>30,1,2))*3+np.where(g.lead<0,0,np.where(g.lead==0,1,2))
            direction=(~g.home_offense.to_numpy()).astype(int);np.add.at(c,(direction,contexts,js),1)
            np.testing.assert_array_equal(c,np.array(raw['transition_counts_flat']).reshape(shape))
            st=np.stack([np.stack([np.bincount(source,weights=c[d,cc],minlength=72) for cc in range(9)]) for d in range(2)])
            np.testing.assert_array_equal(st,np.array(raw['source_counts_flat']).reshape(2,9,72))
            den=st[:,:,source];p=np.divide(c,den,out=np.zeros(shape),where=den>0)
            np.testing.assert_allclose(p,np.array(raw['transition_probabilities_flat']).reshape(shape),atol=1e-7)
            donor=np.stack([teams.get(old[s+'_team_id'],np.zeros((3,e),dtype=np.int64)) for s in ['home','away']])
            td=np.stack([np.stack([np.bincount(source,weights=donor[d,t],minlength=72) for t in range(3)]) for d in range(2)])
            ld=np.stack([np.bincount(source,weights=totals[t],minlength=72) for t in range(3)])
            expected=p.copy();tags=np.where(st>0,0,-1)
            tm=(st==0)&(td[:,times,:]>0);lm=(st==0)&(~tm)&(ld[times,:][None,:,:]>0)
            tags[tm]=1;tags[lm]=2
            tp=np.divide(donor,td[:,:,source],out=np.zeros_like(donor,dtype=float),where=td[:,:,source]>0)[:,times,:]
            lp=np.divide(totals,ld[:,source],out=np.zeros_like(totals,dtype=float),where=ld[:,source]>0)[times,:]
            expected=np.where(tm[:,:,source],tp,expected);expected=np.where(lm[:,:,source],lp[None,:,:],expected)
            np.testing.assert_allclose(expected,np.array(new['transition_probabilities_flat']).reshape(shape),atol=1e-7)
            m=marks.loc[league,gid];np.testing.assert_array_equal(tags,np.array(m.provenance_flat).reshape(2,9,72))
            for side in ['home','away']:assert list(m[side+'_donor_game_ids'])==ids[old[side+'_team_id']]
            assert m.latest_donor_date==latest or (pd.isna(m.latest_donor_date) and latest is None)
            assert latest is None or latest<day
            b=bgroups[league,gid];starts=json.loads(old['half_starts_json'])
            # Independently replay both evidence partitions, rather than builder counts.
            def rewards(z):return (int(np.where(z.home_offense,z.own_points,z.opponent_points).sum()),int(np.where(z.home_offense,z.opponent_points,z.own_points).sum()))
            h,a=rewards(g);bh,ba=rewards(b);h+=bh+sum(x['home_points'] for x in starts);a+=ba+sum(x['away_points'] for x in starts)
            assert (h,a)==(old['home_regulation_score'],old['away_regulation_score'])
            ar=audit.loc[league,gid];assert ar.regulation_home_points==h and ar.regulation_away_points==a and ar.score_evidence_reconciled
            assert raw['segment_count']==len(g) and len(g)+len(b)==old['segment_count']
            pending.append((gid,day,old['home_team_id'],old['away_team_id'],c));checked+=1
        assert count_rows==len(meta)
        print('Verified',league,count_rows,flush=True)
    logs=json.loads((ROOT/'data'/'donor-history-log.json').read_text())
    for log in logs:
        meta=all_meta[log['league']];earlier=meta[meta.day.lt(log['target_date'])&meta.matrix_status.eq('parsed_regulation')]
        assert log['donor_game_ids']==earlier.game_id.tolist()
        assert log['latest_donor_date']==(earlier.day.max() if len(earlier) else None)
    summary=json.loads((ROOT/'data'/'summary.json').read_text())
    for name,h in summary['source_hashes'].items():assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==h
    result=dict(passed=True,rows_verified=rows_checked,parsed_games_verified=checked,null_games_preserved=nulls,
        all_observations_preserved=True,score_evidence_replay_all_games=True,terminal_physical_states=0,
        raw_counts_independently_rebuilt=True,imputed_probabilities_independently_rebuilt=True,donor_dates_and_team_ids_verified=True,
        same_time_bucket_imputation_verified=True,original_source_hashes_unchanged=True,
        limitation='Censored final segments are preserved evidence, not modeled successor transitions. Support/clock realism remains a separate check.')
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':verify()
