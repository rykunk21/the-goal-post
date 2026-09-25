from copy import deepcopy
import numpy as np
import pytest
from goalpost.transitions.simulator import run

def fixture(remaining=9.):
    keys=np.array([[16,17,0,7,0],[16,17,0,3,0]])
    p=np.zeros((2,9,2));p[0,:,0]=1;p[1,:,1]=1
    t={(j,c):[1800-remaining] for j in [0,1] for c in range(9)}
    return keys,p,t

@pytest.mark.parametrize('remaining',[1.,9.,9.999])
def test_under_ten_is_accepted_and_second_half_still_runs(remaining):
    s,d,events,rows=run(*fixture(remaining),n=10)
    np.testing.assert_array_equal(s,np.tile([7,3],(10,1)))
    assert d['completed']==10 and d['grace_halves']==20 and d['clock_expired_halves']==0
    assert d['grace_completed_games']==10 and len(events)==20 and all(r['completed'] for r in rows)
    assert all(e['accepted_under_grace'] for e in events)

@pytest.mark.parametrize('remaining',[10.,10.001,17.,30.])
def test_ten_or_more_is_incomplete(remaining):
    s,d,events,rows=run(*fixture(remaining),n=10)
    assert np.all(s==-1) and d['failed']==10 and d['grace_halves']==0
    assert all(not r['completed'] for r in rows)

def test_disabled_grace_reproduces_missing_failure():
    s,d,_,_=run(*fixture(9),n=10,grace_seconds=0)
    assert np.all(s==-1) and d['failed_missing_rows']==10

def test_expired_clock_is_strict_completion_no_missing_lookup():
    s,d,e,_=run(*fixture(0),n=10)
    assert np.all(s==[7,3]) and d['strict_clock_completed_games']==10 and d['clock_expired_halves']==20
    assert len(e)==0

def test_supported_final_seconds_still_simulated():
    keys=np.array([[16,17,0,7,0],[16,17,0,3,0],[17,17,0,2,0]])
    p=np.zeros((2,9,3));p[0,:,0]=1;p[1,:,1]=1;p[:,:,2]=1
    t={(j,c):[1791. if j<2 else 9.] for j in range(3) for c in range(9)}
    s,d,_,_=run(keys,p,t,n=10)
    assert np.all(s==[9,5]) and d['grace_halves']==0

def test_cap_not_reclassified_even_when_under_ten():
    k,p,t=fixture();k[:,1]=16;t={key:[1795/600] for key in t}
    s,d,_,_=run(k,p,t,n=1)
    assert np.all(s==-1) and d['cap_exhaustions']==1 and d['grace_halves']==0

def test_invalid_inputs_still_raise():
    k,p,t=fixture();p[0,0,0]=float('nan')
    with pytest.raises(ValueError):run(k,p,t,n=1)
    k,p,t=fixture();k[0,1]=-1
    with pytest.raises(ValueError):run(k,p,t,n=1)

def test_state_paired_rerun_preserves_baseline_when_grace_disabled():
    states=[];args=fixture(9);a,da,_,_=run(*args,n=20,grace_seconds=0,capture_states=states)
    b,db,_,_=run(*args,n=20,grace_seconds=0,start_states=states)
    np.testing.assert_array_equal(a,b);assert da==db and len(states)==20

def test_first_half_grace_does_not_turn_later_failure_into_complete_game():
    k,p,t=fixture(9);p[1]=0
    s,d,_,rows=run(k,p,t,n=30)
    assert np.all(s==-1) and d['completed']==0 and d['grace_halves']>0
    assert any(r['used_grace'] and not r['completed'] for r in rows)
