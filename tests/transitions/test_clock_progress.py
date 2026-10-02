import numpy as np
import pandas as pd
import pytest
from goalpost.transitions.clock_progress import coalesce,validate_clock_progress
from goalpost.transitions.simulator import run


def test_penalty_chain_preserves_time_reward_and_raw_evidence():
    a=dict(league='nfl',game_id='g',half=1,offense='A',defense='B',home_offense=True,source=69,destination=69,switch=False,remaining=2.,lead=0,elapsed=0.,own_points=0,opponent_points=0,play_id=1.,next_play_id=2.,raw_rows=2,play_type='no_play')
    b=dict(a,play_id=2.,next_play_id=3.,raw_rows=1)
    c=dict(a,play_id=3.,next_play_id=None,destination=-1,elapsed=2.,own_points=7,raw_rows=3,play_type='pass')
    source=pd.DataFrame([a,b,c]);before=source.copy(deep=True);out,m=coalesce(source)
    assert m==[[0,1,2]] and len(out)==1
    assert out.iloc[0].own_points==7 and out.iloc[0].elapsed==2 and out.iloc[0].raw_rows==6 and out.iloc[0].play_id==1
    pd.testing.assert_frame_equal(source,before)


def test_state_changing_zero_time_penalty_is_retained():
    a=dict(league='nfl',game_id='g',half=1,offense='A',defense='B',home_offense=True,source=69,destination=70,switch=False,remaining=2.,lead=0,elapsed=0.,own_points=0,opponent_points=0,play_id=1.,next_play_id=2.,raw_rows=1,play_type='no_play')
    out,m=coalesce(pd.DataFrame([a]));assert m==[[0]] and out.iloc[0].destination==70


def test_missing_successor_fails_instead_of_dropping_evidence():
    a=dict(source=1,destination=1,switch=False,elapsed=0,own_points=0,opponent_points=0)
    with pytest.raises(ValueError,match='successor'):coalesce(pd.DataFrame([a]))


def test_multistate_zero_time_closed_class_rejected_but_timed_exit_allowed():
    k=np.array([[16,17,0,0,0],[17,16,0,0,0]])
    rows={(0,0,16):(np.array([0]),[1]),(0,0,17):(np.array([1]),[1])}
    t={(0,0):[0],(1,0):[0]}
    with pytest.raises(ValueError,match='Zero-clock'):validate_clock_progress(k,rows,t)
    t[1,0]=[0,1];validate_clock_progress(k,rows,t)


def test_simulator_refuses_zero_time_self_loop_before_sampling():
    k=np.array([[16,16,0,0,0]]);p=np.ones((2,9,1));t={(0,c):[0] for c in range(9)}
    with pytest.raises(ValueError,match='Zero-clock'):run(k,p,t,n=1)
