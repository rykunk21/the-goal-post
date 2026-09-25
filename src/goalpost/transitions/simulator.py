"""Owner-authorized missing-row completion approximation, strictly <10 seconds.

At a missing probability row only: retain score so far and accept that half
under the grace rule. Always continue the second half. Never convert errors,
invalid probabilities or loop caps into completed games. No extra points.
"""
from copy import deepcopy
import numpy as np

def context(rem,lead):return (0 if rem>120 else 1 if rem>30 else 2)*3+(0 if lead<0 else 1 if lead==0 else 2)

def run(keys,p,records,n=50000,seed=20260924,grace_seconds=10.,start_states=None,capture_states=None):
    if grace_seconds not in (0,10):raise ValueError('Registered thresholds: 0 or 10 seconds')
    if start_states is not None and len(start_states)!=n:raise ValueError('RNG state count mismatch')
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
    events=[];outcomes=[]
    diag=dict(grace_seconds=grace_seconds,strict_clock_completed_games=0,grace_completed_games=0,grace_halves=0,grace_seconds_forgone_sum=0.,failed_missing_rows=0,attempted=n,completed=0,failed=0,missing_probability_rows=0,cap_exhaustions=0,
        missing_row_counts={},missing_row_examples=[],sampled_segments=0,timing_same_bucket_fallbacks=0,clock_censored_segments=0,clock_censor_seconds_sum=0.,
        clock_expired_halves=0,artificial_terminal_stops=0,probability_runtime_fallbacks=0)
    for i in range(n):
        if start_states is not None:rng.bit_generator.state=deepcopy(start_states[i])
        if capture_states is not None:capture_states.append(deepcopy(rng.bit_generator.state))
        used_grace=False;failure_reason=None
        home=away=0;failed=False;starter=int(rng.integers(2))
        for half in range(2):
            d=starter if half==0 else 1-starter;s=16;remaining=1800.;half_grace=False
            for step in range(600):
                if remaining<=0:break
                c=context(remaining,(home-away)*(1 if d==0 else -1));row=rows.get((d,c,s))
                if row is None:
                    diag['missing_probability_rows']+=1
                    key=f'{d},{c},{s}';diag['missing_row_counts'][key]=diag['missing_row_counts'].get(key,0)+1
                    if len(diag['missing_row_examples'])<10:diag['missing_row_examples'].append(dict(simulation=i,half=half+1,direction=d,context=c,state=s,remaining=remaining,home_score=home,away_score=away))
                    accepted=0<remaining<grace_seconds
                    events.append(dict(simulation=i,half=half+1,direction=d,context=c,state=s,remaining=remaining,home_score=home,away_score=away,accepted_under_grace=accepted))
                    if accepted:
                        used_grace=True;half_grace=True;diag['grace_halves']+=1
                        diag['grace_seconds_forgone_sum']+=remaining
                    else:
                        failed=True;failure_reason='missing_probability_row';diag['failed_missing_rows']+=1
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
                if remaining>0:diag['cap_exhaustions']+=1;failed=True;failure_reason='segment_cap'
            if failed:break
            if half_grace:continue
            if remaining>0:raise AssertionError('Half stopped before clock expiration')
            diag['clock_expired_halves']+=1
        if failed:diag['failed']+=1
        else:
            scores[i]=home,away;diag['completed']+=1
            diag['grace_completed_games' if used_grace else 'strict_clock_completed_games']+=1
        outcomes.append(dict(simulation=i,completed=not failed,used_grace=used_grace,failure_reason=failure_reason,home_score=int(scores[i,0]),away_score=int(scores[i,1])))
    return scores,diag,events,outcomes
