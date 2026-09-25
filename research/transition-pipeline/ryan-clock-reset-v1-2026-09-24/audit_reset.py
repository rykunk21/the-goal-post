"""Read-only assessment of the proposed terminal-destination reset."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'ryan-game-tables-imputation-v1-2026-09-24'/'data'
keys=np.asarray(json.loads((DATA/'transition_set.json').read_text())['keys'])
s=pd.read_parquet(DATA/'parsed_segments.parquet')
s['context']=np.where(s.remaining>120,0,np.where(s.remaining>30,1,2))*3+np.where(s.lead<0,0,np.where(s.lead==0,1,2))
s['terminal']=s.destination.eq(-1)
report={'status':'preflight_only_clock_design_pending','source_sha256':{n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['transition_set.json','parsed_segments.parquet']},
        'catalog':{'edges':len(keys),'terminal_destination_edges':int((keys[:,1]==-1).sum()),'terminal_source_edges':int((keys[:,0]==-1).sum()),'physical_source_states':len(set(keys[:,0].tolist()))},
        'clock_representation':'Source context only; destination-clock bucket is not a key field. Elapsed time is in a separate observed-segment table.',
        'leagues':{}}
for league,g in s.groupby('league'):
    t=g[g.terminal]
    groups=g.groupby(['game_id','home_offense','context','source']).terminal.agg(['size','sum'])
    report['leagues'][league]={'parsed_games':g.game_id.nunique(),'segments':len(g),'terminal_segments':len(t),
        'scoring_terminal_segments':int((t.own_points+t.opponent_points>0).sum()),
        'points_lost_if_terminal_segments_deleted':int((t.own_points+t.opponent_points).sum()),
        'observed_conditional_rows_made_empty_by_deletion':int((groups['size']==groups['sum']).sum()),
        'terminal_elapsed_equals_remaining':bool(np.allclose(t.elapsed,t.remaining)),
        'terminal_segments_with_historical_start_over_120_seconds':int(t.remaining.gt(120).sum())}
(ROOT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
