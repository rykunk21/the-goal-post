"""Stream game rows back; verify probabilities, count/reward replay and schema."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import pyarrow.parquet as pq


def verify(root):
    root=Path(root);catalog=json.loads((root/'transition_set.json').read_text())
    keys=np.asarray(catalog['keys'],dtype=np.int32);sources=keys[:,0]
    schemas=[pq.read_schema(root/f'{league}_games.parquet') for league in ('nfl','college')]
    assert schemas[0].equals(schemas[1],check_metadata=True)
    counts_by_game={}
    # Independently aggregate saved event rewards and number of segments.
    for batch in pq.ParquetFile(root/'parsed_segments.parquet').iter_batches(batch_size=4096):
        for e in batch.to_pylist():
            k=(e['league'],e['game_id']);v=counts_by_game.setdefault(k,[0,0,0])
            v[0]+=e['own_points'] if e['home_offense'] else e['opponent_points']
            v[1]+=e['opponent_points'] if e['home_offense'] else e['own_points']
            v[2]+=1
    report={}
    for league in ('nfl','college'):
        ids=set();statuses=Counter();fcs=Counter()
        for batch in pq.ParquetFile(root/f'{league}_games.parquet').iter_batches(batch_size=8):
            for r in batch.to_pylist():
                assert r['game_id'] not in ids;ids.add(r['game_id']);statuses[r['matrix_status']]+=1
                assert r['game_date'] and r['home_team_id']!=r['away_team_id']
                assert r['transition_set_version']==catalog['version']
                assert r['matrix_shape']==catalog['shape']
                if r['home_division']=='fcs' and r['away_division']=='fcs':fcs[r['matrix_status']]+=1
                if r['matrix_status']!='parsed_regulation':
                    assert all(r[n] is None for n in ['transition_probabilities_flat','transition_counts_flat','source_counts_flat','half_starts_json'])
                    assert r['matrix_issue'];continue
                p=np.asarray(r['transition_probabilities_flat']).reshape(r['matrix_shape'])
                c=np.asarray(r['transition_counts_flat']).reshape(r['matrix_shape'])
                t=np.asarray(r['source_counts_flat']).reshape(2,9,72)
                assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
                assert (c>=0).all() and np.equal(c,c.astype(np.int64)).all()
                assert c.sum()==t.sum()==r['segment_count']
                expected=np.divide(c,t[...,sources],out=np.zeros_like(p),where=t[...,sources]>0)
                assert np.allclose(p,expected,rtol=1e-6,atol=1e-7)
                for source in range(72):
                    ix=sources==source
                    assert np.allclose(p[...,ix].sum(-1),t[...,source]>0,atol=1e-6)
                openings=json.loads(r['half_starts_json']);assert len(openings)==2
                opening=np.array([sum(s['home_points'] for s in openings),sum(s['away_points'] for s in openings)])
                rewards=c[0].sum(0)@keys[:,3:5]+(c[1].sum(0)@keys[:,3:5])[::-1]+opening
                assert tuple(rewards)==(r['home_regulation_score'],r['away_regulation_score'])
                ev=counts_by_game[(league,r['game_id'])]
                assert ev[2]==r['segment_count'] and np.array_equal(np.array(ev[:2])+opening,rewards)
                assert r['home_final_score']>=r['home_regulation_score']
                assert r['away_final_score']>=r['away_regulation_score']
        report[league]=dict(rows=len(ids),matrix_statuses=dict(statuses),standalone_fcs=dict(fcs))
    result=dict(passed=True,identical_table_schemas=True,unique_game_rows=True,
                all_exported_matrix_counts_replay_regulation_scores=True,
                missing_or_failed_matrices_are_null=True,tables=report,
                limitation='Exact count/reward reconciliation is not Monte Carlo calibration or demonstrated predictive skill.')
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);verify(p.parse_args().data)
