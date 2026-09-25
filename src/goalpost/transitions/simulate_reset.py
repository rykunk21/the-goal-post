"""Clock-only regulation diagnostic. No terminal physical state or stop edge.

Rewards are at segment completion, retaining the conservative existing clock
approximation: censor a segment that cannot fit in remaining time. This is not
an exact NFL final-snap/conversion model. Boundary-censored source observations
are not sampled as reusable football transitions.
"""
from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib
import numpy as np
import pandas as pd
from .paths import RESET as ROOT
OUT=ROOT/'matchup'
N=50000;SEED=20260924

def context(rem,lead):return (0 if rem>120 else 1 if rem>30 else 2)*3+(0 if lead<0 else 1 if lead==0 else 2)
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def prepare(folder):
    data=ROOT/folder;cat=json.loads((data/'transition_set.json').read_text());keys=np.array(cat['keys'],dtype=int)
    if np.any(keys[:,:2]<0):raise ValueError('Terminal state forbidden')
    cols=['game_id','game_date','home_team_id','away_team_id','season','matrix_status','matrix_shape','transition_probabilities_flat']
    meta=pd.read_parquet(data/'nfl_games.parquet',columns=cols[:-1]);blends=[];details={}
    for team in ['GB','ATL']:
        select=meta[meta.season.eq(2026)&meta.matrix_status.eq('parsed_regulation')&(meta.home_team_id.eq(team)|meta.away_team_id.eq(team))].sort_values(['game_date','game_id'])
        rows=pd.read_parquet(data/'nfl_games.parquet',columns=cols,filters=[('game_id','in',select.game_id.tolist())]).set_index('game_id')
        ps=[]
        for gid in select.game_id:
            r=rows.loc[gid];p=np.array(r.transition_probabilities_flat).reshape(r.matrix_shape);ps.append(p[0 if r.home_team_id==team else 1])
        blend=np.zeros_like(ps[0]);contributors=Counter()
        for s in range(72):
            ix=np.flatnonzero(keys[:,0]==s)
            for c in range(9):
                supported=[p[c,ix] for p in ps if p[c,ix].sum()>0];contributors[len(supported)]+=1
                if supported:
                    if not all(np.isclose(v.sum(),1,atol=1e-6) for v in supported):raise ValueError('Bad probability row')
                    v=np.mean(supported,axis=0);blend[c,ix]=v/v.sum()
        blends.append(blend);details[team]={'games':select[['game_id','game_date','home_team_id','away_team_id']].to_dict('records'),'rows_by_contributing_games':dict(contributors)}
    seg=pd.read_parquet(data/'parsed_segments.parquet');seg=seg[seg.league.eq('nfl')]
    index={tuple(k):j for j,k in enumerate(keys)};records=defaultdict(list)
    for r in seg.itertuples():
        j=index[(r.source,r.destination,int(r.switch),r.own_points,r.opponent_points)];c=context(r.remaining,r.lead)
        records[j,c].append(float(r.elapsed));records[j,9+c//3].append(float(r.elapsed))
    return keys,np.array(blends),records,details

def run(keys,p,records,n=N,seed=SEED):
    if np.any(keys[:,:2]<0) or np.any(keys[:,:2]>=72):raise ValueError('Non-football state forbidden')
    if not np.isfinite(p).all() or np.any(p<0):raise ValueError('Invalid probabilities')
    rows={};timing={};fallback=set()
    for d in range(2):
        for s in range(72):
            ix=np.flatnonzero(keys[:,0]==s)
            for c in range(9):
                values=p[d,c,ix];positive=values>0
                if positive.any():
                    if not np.isclose(values.sum(),1,atol=1e-6):raise ValueError('Bad probability normalization')
                    js=ix[positive];cdf=np.cumsum(values[positive]/values.sum());cdf[-1]=1;rows[d,c,s]=(js,cdf)
                    for j in js:
                        if (j,c) not in timing:
                            if records.get((int(j),c)):times=records[int(j),c]
                            elif records.get((int(j),9+c//3)):
                                times=records[int(j),9+c//3];fallback.add((int(j),c))
                            else:raise ValueError('No timing evidence in the same time bucket')
                            if not np.isfinite(times).all() or np.any(np.array(times)<0):raise ValueError('Invalid duration')
                            timing[j,c]=times
    rng=np.random.default_rng(seed);scores=np.full((n,2),-1,dtype=np.int32)
    diag=dict(attempted=n,completed=0,failed=0,missing_probability_rows=0,cap_exhaustions=0,
        missing_row_counts={},missing_row_examples=[],sampled_segments=0,timing_same_bucket_fallbacks=0,clock_censored_segments=0,clock_censor_seconds_sum=0.,
        clock_expired_halves=0,artificial_terminal_stops=0,probability_runtime_fallbacks=0)
    for i in range(n):
        home=away=0;failed=False;starter=int(rng.integers(2))
        for half in range(2):
            d=starter if half==0 else 1-starter;s=16;remaining=1800.
            for step in range(600):
                if remaining<=0:break
                c=context(remaining,(home-away)*(1 if d==0 else -1));row=rows.get((d,c,s))
                if row is None:
                    diag['missing_probability_rows']+=1;failed=True
                    key=f'{d},{c},{s}';diag['missing_row_counts'][key]=diag['missing_row_counts'].get(key,0)+1
                    if len(diag['missing_row_examples'])<10:diag['missing_row_examples'].append(dict(simulation=i,half=half+1,direction=d,context=c,state=s,remaining=remaining,home_score=home,away_score=away))
                    break
                js,cdf=row;j=int(js[np.searchsorted(cdf,rng.random(),side='right')]);times=timing[j,c]
                duration=times[int(rng.integers(len(times)))];diag['timing_same_bucket_fallbacks']+=int((j,c) in fallback)
                if duration>remaining:
                    diag['clock_censored_segments']+=1;diag['clock_censor_seconds_sum']+=remaining;remaining=0.;break
                _,destination,switch,own,opp=map(int,keys[j])
                if d==0:home+=own;away+=opp
                else:home+=opp;away+=own
                remaining-=duration;diag['sampled_segments']+=1;s=destination
                if switch:d=1-d
            else:
                if remaining>0:diag['cap_exhaustions']+=1;failed=True
            if failed:break
            if remaining>0:raise AssertionError('Half stopped before clock expiration')
            diag['clock_expired_halves']+=1
        if failed:diag['failed']+=1
        else:scores[i]=home,away;diag['completed']+=1
    return scores,diag

def summarize(scores,diag):
    valid=scores[(scores>=0).all(1)];r={'n':len(valid),'diagnostics':diag,'conditional_on_completion':len(valid)!=len(scores)}
    if len(valid):
        r.update(home_mean=float(valid[:,0].mean()),away_mean=float(valid[:,1].mean()),distributions={})
        for name,v in [('total',valid.sum(1)),('home_minus_away',valid[:,0]-valid[:,1])]:
            r['distributions'][name]=dict(mu=float(v.mean()),sigma=float(v.std()),median=float(np.median(v)),q05=float(np.quantile(v,.05)),q95=float(np.quantile(v,.95)))
    return r

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    settings={'home':'Green Bay Packers','away':'Atlanta Falcons','n':N,'seed':SEED,'probability_blend':'Equal weights per supported row across two 2026 own-offense matrices per team',
        'clock':'Two 1800-second halves; elapsed time from observed NFL segments, exact context or same time bucket. No terminal stop edge.',
        'imputation':'Earlier UTC dates; team then league within same clock bucket, pooled lead categories',
        'starting_state':16,'first_possession':'Fair coin; opposite team opens second half','regulation_only':True,'home_field_adjustment':None,'opponent_defense_adjustment':None,
        'score_labels_loaded':False,'boundary_evidence_loaded_for_simulation':False,
        'limitations':['Boundary-censored last segments excluded from transition fit, rewards preserved only in evidence sidecar.',
            'A segment that exceeds remaining time is censored without scoring; exact last-play/conversion timing unresolved.',
            'Same empirical team offense blend, not VAE-generated or defense-adjusted. Descriptive global catalog, no predictive validation.'],
        'source_hashes':{n:hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest() for n in ['nfl_games.parquet','transition_set.json','parsed_segments.parquet']}}
    results={}
    for label,folder in [('raw','raw'),('imputed','data')]:
        keys,p,t,details=prepare(folder);settings[label+'_teams']=details
        scores,diag=run(keys,p,t);results[label]=summarize(scores,diag);np.save(OUT/f'{label}-scores.npy',scores)
        np.savez_compressed(OUT/f'{label}-kernels.npz',keys=keys,probabilities=p)
        print(label,json.dumps(results[label]),flush=True)
    write('settings.json',settings);write('results.json',results)
    valid=np.load(OUT/'imputed-scores.npy');valid=valid[(valid>=0).all(1)]
    for name,v in [('total',valid.sum(1)),('home_minus_away',valid[:,0]-valid[:,1])]:
        xs,counts=np.unique(v,return_counts=True);pd.DataFrame({'points':xs,'count':counts,'probability':counts/len(v)}).to_csv(OUT/f'{name}-pmf.csv',index=False)
if __name__=='__main__':main()
