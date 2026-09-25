import copy
import pytest
from test_tables import espn_fixture
from raw_espn_college import summary_frame, extract_raw_espn


def raw_fixture():
    frame, meta = espn_fixture()
    payload = {'header': {'id': meta['game_id'], 'competitions': [{
        'date': meta['game_date'], 'status': {'type': {'completed': True}},
        'competitors': [{'homeAway': side, 'id': meta[side+'_team_id'],
                         'score': str(meta[side+'_final_score'])} for side in ('home','away')]}]},
        'drives': {'previous': [{'plays': []}, {'plays': []}]}}
    for i, row in frame.iterrows():
        play={'id': str(1000+i), 'sequenceNumber': str(i % 4)}
        for key in ('homeScore','awayScore','period.number','type.text','start.down',
                    'start.team.id','start.distance','start.yardsToEndzone','clock.displayValue'):
            target=play;parts=key.split('.')
            for part in parts[:-1]:target=target.setdefault(part,{})
            target[parts[-1]]=row[key]
        payload['drives']['previous'][int(i>=7)]['plays'].append(play)
    return payload,meta


def test_array_order_and_exact_id_mapping_without_global_sequence_sort():
    raw, meta=raw_fixture();before=copy.deepcopy(raw)
    raw['drives']['previous'][0]['plays'][1]['id']='900000000000000001'
    frame,mapping=summary_frame(raw,meta)
    game,mapping=extract_raw_espn(raw,meta)
    assert game.final==game.regulation==(14,0) and len(game.events)==5
    assert mapping[1]['source_play_id']=='900000000000000001'
    assert [m['canonical_play_id'] for m in mapping]==list(range(1,12))
    assert mapping[7]['drive_array_index']==1 and mapping[7]['play_array_index']==0
    assert raw['drives']==dict(previous=raw['drives']['previous'])
    # The adapter must not mutate its input.
    before=copy.deepcopy(raw);extract_raw_espn(raw,meta);assert raw==before


@pytest.mark.parametrize('mutation,match',[
    ('event','event identity'),('team','team identity'),('score','final mismatch'),
    ('date','date mismatch'),('incomplete','incomplete'),('duplicate','duplicate'),
    ('numeric_id','exact play ID'),('empty','no drive'),('current','current drive'),
    ('period','period order'),('clock','nonmonotonic'),('invalid_clock','invalid clock'),
    ('score_reversal','scoreboard reversal'),('unknown_type','unreviewed'),
    ('possession','possession'),
])
def test_corrupt_source_rejected(mutation,match):
    raw,meta=raw_fixture();comp=raw['header']['competitions'][0]
    plays=raw['drives']['previous'][0]['plays']
    if mutation=='event':raw['header']['id']='other'
    elif mutation=='team':comp['competitors'][0]['id']='wrong'
    elif mutation=='score':comp['competitors'][0]['score']='21'
    elif mutation=='date':comp['date']='2020-01-01'
    elif mutation=='incomplete':comp['status']['type']['completed']=False
    elif mutation=='duplicate':plays[1]['id']=plays[0]['id']
    elif mutation=='numeric_id':plays[1]['id']=1001
    elif mutation=='empty':raw['drives']['previous']=[]
    elif mutation=='current':raw['drives']['current']={'plays': [plays[0]]}
    elif mutation=='period':plays[3]['period']['number']=2
    elif mutation=='clock':plays[2]['clock']['displayValue']='14:59'
    elif mutation=='invalid_clock':plays[1]['clock']['displayValue']='15:01'
    elif mutation=='score_reversal':plays[3]['homeScore']=0
    elif mutation=='unknown_type':plays[1]['type']['text']='mystery'
    elif mutation=='possession':plays[1]['start']['team']['id']=99
    with pytest.raises((ValueError,TypeError),match=match):extract_raw_espn(raw,meta)


@pytest.mark.parametrize('gid,error', [('401868124',None),
    ('401866532','nonmonotonic/missing half clock'),('401866615','scoreboard reversal')])
def test_real_fcs_regressions(gid,error):
    import json
    from pathlib import Path
    evidence=json.loads((Path(__file__).parent/'fixtures'/f'{gid}.json').read_text())
    if error:
        with pytest.raises(ValueError,match=error):
            extract_raw_espn(evidence['summary'],evidence['metadata'])
    else:
        g,mapping=extract_raw_espn(evidence['summary'],evidence['metadata'])
        assert g.final==(evidence['metadata']['home_final_score'],evidence['metadata']['away_final_score'])
        assert len(g.events)>50 and len(mapping)>len(g.events)
