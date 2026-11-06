"""Coarse-state zero-time reduction and preflight clock-progress validation."""
import numpy as np
import pandas as pd


def coalesce(segments):
    """Fold zero-time/no-reward self segments into their observed successor.

    Penalty evidence remains in the source and the index map. This operates on
    the declared coarse state, not an assertion that raw yardage was unchanged.
    """
    rows=segments.to_dict('records');output=[];mapping=[];i=0
    while i<len(rows):
        first=rows[i];last=first;positions=[i];i+=1
        while (last['destination']==last['source'] and not last['switch']
               and last['elapsed']==0 and last['own_points']==0 and last['opponent_points']==0):
            if i==len(rows):raise ValueError('Zero-time segment has no observed successor')
            nxt=rows[i]
            for field in ['league','game_id','half','offense','defense','home_offense','source','remaining','lead']:
                if last[field]!=nxt[field]:raise ValueError('Zero-time successor mismatch: '+field)
            if last['next_play_id']!=nxt['play_id']:raise ValueError('Zero-time successor identity mismatch')
            positions.append(i);last=nxt;i+=1
        merged=dict(last);merged['play_id']=first['play_id'];merged['raw_rows']=sum(rows[j]['raw_rows'] for j in positions)
        output.append(merged);mapping.append(positions)
    return pd.DataFrame(output,columns=segments.columns),mapping


def validate_clock_progress(keys, rows, timing):
    """Reject closed zero-duration classes for each fixed clock/lead context.

    Nodes include possession direction. Zero rewards keep the score context
    fixed; positive-time or scoring exits remove a node from the trapped set.
    Conservative across all supported states, including potentially unreachable
    ones. Runtime cap remains necessary for other forms of nonprogress.
    """
    for ctx in range(9):
        successors={}
        for (direction,c,state),(js,cdf) in rows.items():
            if c!=ctx:continue
            dest=set();eligible=True
            for j in js:
                _,end,switch,own,opp=map(int,keys[j])
                if own or opp or np.any(np.asarray(timing[j,c])>0):eligible=False;break
                dest.add((1-direction if switch else direction,end))
            if eligible:successors[direction,state]=dest
        trapped=set(successors)
        while True:
            keep={node for node in trapped if successors[node] and successors[node]<=trapped}
            if keep==trapped:break
            trapped=keep
        if trapped:
            raise ValueError(f'Zero-clock closed transition class in context {ctx}: {sorted(trapped)}; rebuild corrected data')
