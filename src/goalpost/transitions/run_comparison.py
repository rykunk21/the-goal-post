"""Paired comparison preserving every original baseline game's random start."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import pandas as pd
from .simulator import run
from .paths import GRACE as ROOT, RESET as BASE
from . import simulate_reset as base

def save(name,x):(ROOT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    keys,p,t,details=base.prepare('data');states=[]
    strict,ds,es,rs=run(keys,p,t,grace_seconds=0,capture_states=states)
    np.testing.assert_array_equal(strict,np.load(BASE/'matchup/imputed-scores.npy'))
    print('Strict baseline exactly reproduced:',ds['completed'],flush=True)
    tolerant,dt,et,rt=run(keys,p,t,grace_seconds=10,start_states=states)
    sv=(strict>=0).all(1);tv=(tolerant>=0).all(1)
    assert np.all(tv[sv]);np.testing.assert_array_equal(strict[sv],tolerant[sv])
    assert sum(r['completed'] for r in rt)==int(tv.sum())
    newly=~sv&tv
    comparison=dict(attempted=len(strict),strict_completed=int(sv.sum()),grace_rule_completed=int(tv.sum()),
        newly_included_games=int(newly.sum()),still_failed=int((~tv).sum()),strict_completed_scores_unchanged=True,
        baseline_exactly_reproduced=True,threshold='0 < remaining half-clock < 10 seconds, only on a missing probability row',
        approximation='Keep accumulated score; no unsampled final-play reward. First-half acceptance still requires simulating the second half.',
        rng_method='Capture each original strict-run RNG start; replay both policies from the same per-game state. Additional first-half draws cannot perturb later games.')
    np.save(ROOT/'scores.npy',tolerant)
    pd.DataFrame(rt).to_parquet(ROOT/'simulation-outcomes.parquet',index=False)
    pd.DataFrame(et).to_parquet(ROOT/'missing-row-events.parquet',index=False)
    pd.DataFrame(es).to_parquet(ROOT/'baseline-missing-row-events.parquet',index=False)
    summaries={'strict':base.summarize(strict,ds),'under_10_seconds':base.summarize(tolerant,dt),
        'newly_included':base.summarize(tolerant[newly],{'attempted':int(newly.sum()),'completed':int(newly.sum())})}
    save('comparison.json',comparison);save('results.json',summaries)
    save('settings.json',dict(teams=details,seed=20260924,n=50000,grace_seconds=10,
        default_applies_to='Missing-row termination only, independently at each half',
        source_files={(str(f.relative_to(BASE)) if f.is_relative_to(BASE) else f.name):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(base.__file__),BASE/'data/transition_set.json',BASE/'data/nfl_games.parquet',BASE/'data/parsed_segments.parquet',BASE/'matchup/imputed-scores.npy']},
        dataset_changed=False,live_engine_changed=False))
    valid=tolerant[tv]
    for name,v in [('total',valid.sum(1)),('home_minus_away',valid[:,0]-valid[:,1])]:
        x,c=np.unique(v,return_counts=True);pd.DataFrame(dict(points=x,count=c,probability=c/len(valid))).to_csv(ROOT/f'{name}-pmf.csv',index=False)
    print(json.dumps(comparison,indent=2),flush=True)
    print(json.dumps(summaries['under_10_seconds']['distributions'],indent=2),flush=True)
if __name__=='__main__':main()
