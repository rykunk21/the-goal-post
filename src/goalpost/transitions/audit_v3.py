"""Compare v3 against its frozen predecessor and expose clock limitations."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd


def audit(previous,current):
    before=pd.read_parquet(previous/'parsed_segments.parquet')
    after=pd.read_parquet(current/'parsed_segments.parquet')
    cols=before.columns.tolist();sort=['league','game_id','play_id']
    ids=set(zip(before.league,before.game_id))
    keep=np.array([(l,g) in ids for l,g in zip(after.league,after.game_id)])
    pd.testing.assert_frame_equal(before.sort_values(sort).reset_index(drop=True),
        after.loc[keep,cols].sort_values(sort).reset_index(drop=True),check_dtype=False)
    oldkeys=json.loads((previous/'transition_set.json').read_text())['keys']
    newkeys=json.loads((current/'transition_set.json').read_text())['keys']
    assert set(map(tuple,oldkeys)) <= set(map(tuple,newkeys))
    fields=['game_id','matrix_status','pbp_source','home_team_id','away_team_id','game_date',
            'home_final_score','away_final_score','home_division','away_division','season']
    result={};quality=[]
    for league in ('nfl','college'):
        old=pd.read_parquet(previous/f'{league}_games.parquet',columns=fields).set_index('game_id')
        new=pd.read_parquet(current/f'{league}_games.parquet',columns=fields).set_index('game_id')
        pd.testing.assert_index_equal(old.index,new.index)
        parsed=old.matrix_status.eq('parsed_regulation')
        pd.testing.assert_frame_equal(old.loc[parsed],new.loc[parsed])
        added=new[~parsed & new.matrix_status.eq('parsed_regulation')]
        result[league]={'previous_parsed':int(parsed.sum()),'current_parsed':int(new.matrix_status.eq('parsed_regulation').sum()),
                       'added_game_ids':added.index.tolist(),'previous_parsed_segments_unchanged':True,
                       'previous_sources_and_metadata_unchanged':True}
    for (league,gid),g in after.groupby(['league','game_id']):
        streak=maximum=0
        for elapsed in g.sort_values(['half','play_id']).elapsed:
            streak=streak+1 if elapsed==0 else 0;maximum=max(maximum,streak)
        quality.append(dict(league=league,game_id=gid,segment_count=len(g),
                            zero_elapsed_segments=int(g.elapsed.eq(0).sum()),
                            zero_elapsed_fraction=float(g.elapsed.eq(0).mean()),
                            max_consecutive_zero_elapsed_segments=maximum,
                            max_elapsed_seconds=float(g.elapsed.max()),
                            interpretation='Descriptive clock diagnostic; zero elapsed may be legitimate or stale. Not clock-model approval.'))
    pd.DataFrame(quality).to_parquet(current/'clock_quality.parquet',index=False,compression='zstd')
    (current/'change_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('previous',type=Path);p.add_argument('current',type=Path)
    a=p.parse_args();audit(a.previous,a.current)
