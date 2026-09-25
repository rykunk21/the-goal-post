import numpy as np
import pytest
from goalpost.transitions.build_reset import fill,History,support
from goalpost.transitions.simulate_reset import run

def toy():
    k=np.array([[16,16,0,7,0],[16,16,0,3,0]])
    p=np.zeros((2,9,2));p[0,:,0]=1;p[1,:,1]=1
    times={(j,c):[1800.] for j in [0,1] for c in range(9)}
    return k,p,times

def test_clock_only_termination_home_away_orientation():
    s,d=run(*toy(),n=20);assert np.all(s==[7,3]);assert d['clock_expired_halves']==40
    assert d['artificial_terminal_stops']==0 and d['failed']==0

def test_old_terminal_keys_rejected():
    k,p,t=toy();k[0,1]=-1
    with pytest.raises(ValueError,match='Non-football'):run(k,p,t,n=1)

def test_defensive_points_and_score_margin():
    k,p,t=toy();k[0,3:]=[0,2];s,d=run(k,p,t,n=20)
    assert np.all(s==[0,5]) and np.all(s[:,0]-s[:,1]==-5)

def test_repeated_transitions_continue_until_clock():
    k,p,t=toy();t={key:[900.] for key in t};s,d=run(k,p,t,n=20)
    assert np.all(s==[14,6]) and d['sampled_segments']==80

def test_possession_switch_and_defensive_attribution():
    k,p,t=toy();k[:,2]=1;t={key:[900.] for key in t};s,d=run(k,p,t,n=20)
    assert np.all(s==[14,6])

def test_missing_rows_fail_not_zero_score_completion():
    k,p,t=toy();p[0]=0;s,d=run(k,p,t,n=20)
    assert np.all(s==-1) and d['failed']==20 and d['missing_probability_rows']==20

def test_zero_time_loop_fails_cap():
    k,p,t=toy();t={key:[0.] for key in t};s,d=run(k,p,t,n=1)
    assert np.all(s==-1) and d['cap_exhaustions']==1

def test_existing_conservative_clock_censorship_is_explicit():
    k,p,t=toy();t={key:[1801.] for key in t};s,d=run(k,p,t,n=1)
    assert np.all(s==0) and d['clock_censored_segments']==2 and d['clock_expired_halves']==2

def test_timing_fallback_cannot_cross_time_bucket():
    k,p,t=toy();t={(j,10):[1800.] for j in [0,1]}
    with pytest.raises(ValueError,match='same time bucket'):run(k,p,t,n=1)

def test_probability_imputation_same_time_and_preserves_observed_zeros():
    k,p,_=toy();counts=np.zeros_like(p,dtype=int);counts[0,0]=[1,0];p[:]=0;p[0,0,0]=1
    team=np.zeros((2,3,2));team[:,0,1]=5;league=np.zeros((3,2));league[1,0]=3
    before=p.copy();out,marks=fill(p,counts,k,team,league)
    assert np.array_equal(out[0,0],[1,0]);assert np.array_equal(out[0,1],[0,1])
    assert np.array_equal(out[0,3],[1,0]);assert out[0,6].sum()==0
    assert marks[0,1,16]==1 and marks[0,3,16]==2 and marks[0,6,16]==-1
    np.testing.assert_array_equal(p,before)

def test_history_uses_observed_counts_and_team_identity():
    h=History(2);c=np.zeros((2,9,2),dtype=int);c[0,2,0]=2;c[1,7,1]=4
    h.add('g','2026-01-01','A','B',c)
    assert h.team('A')[0,0]==2 and h.team('A')[2].sum()==0
    assert h.team('B')[2,1]==4 and h.ids['B']==['g']
    with pytest.raises(ValueError):h.add('earlier','2025-01-01','A','B',c)

def test_seed_and_preserved_input():
    k,p,t=toy();before=p.copy();a,da=run(k,p,t,n=20,seed=3);b,db=run(k,p,t,n=20,seed=3)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(p,before);assert da==db
