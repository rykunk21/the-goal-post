"""Standalone build integration, publication and failure boundaries."""
import json
from pathlib import Path
import pandas as pd
import pytest
from goalpost.transitions import build_artifacts as builder
from goalpost.transitions.build_tables import CFB_COLUMNS
from goalpost.transitions.weekly import Dataset


@pytest.fixture
def sources(tmp_path):
    root = tmp_path/'sources'
    root.mkdir()
    pd.DataFrame([dict(game_id='toy', season=2026, week=1, game_type='REG',
                       gameday='2026-09-01', home_team='GB', away_team='ATL',
                       home_score=0, away_score=0)]).to_csv(root/'nfl-schedule.csv', index=False)
    plays = pd.DataFrame([dict(game_id='toy', week=1, game_date='2026-09-01',
        home_team='GB', away_team='ATL', home_score=0, away_score=0,
        play_id=i+1, qtr=q, posteam=team, down=1, ydstogo=10, yardline_100=50,
        play_type='run', half_seconds_remaining=clock, posteam_score_post=0,
        defteam_score_post=0, total_home_score=0, total_away_score=0)
        for i,(q,team,clock) in enumerate([(1,'GB',1800),(1,'ATL',900),(3,'ATL',1800),(3,'GB',900)])])
    college_schedule = pd.DataFrame(columns=['game_id','completed','home_points','away_points',
        'home_division','away_division','start_date'])
    for year in range(2023, 2027):
        name = 'play_by_play_2023.parquet' if year == 2023 else f'nfl-{year}.parquet'
        (plays if year == 2026 else plays.iloc[:0]).to_parquet(root/name, index=False)
        college_schedule.to_parquet(root/f'cfb_schedules-{year}.parquet', index=False)
        pd.DataFrame(columns=CFB_COLUMNS).to_parquet(root/f'cfb-{year}.parquet', index=False)
    return root


def test_real_build_and_weekly_load_without_environment(sources, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GOALPOST_TRANSITION_DATA', str(tmp_path/'wrong-root'))
    output = builder.build(sources)
    assert output == tmp_path/'artifacts/transitions'
    assert not (tmp_path/'wrong-root').exists()
    report = json.loads((output/'build-report.json').read_text())
    assert report['status'] == 'complete'
    assert report['steps'] == ['extract','verify_extracted','reset_and_impute','verify_reset']
    assert len(report['source_hashes']) == 13
    for path, digest in report['artifact_hashes'].items():
        assert builder.sha256(output/path) == digest
    assert json.loads((output/'reset/data/summary.json').read_text())['raw_source'] == str(output/'extracted')
    data = Dataset(output/'reset/data')
    assert data.meta.game_id.tolist() == ['toy']
    assert data.meta.matrix_status.tolist() == ['parsed_regulation']
    assert (data.keys[:,:2] >= 0).all()
    with pytest.raises(ValueError, match='overwrite'):
        builder.build(sources)


def test_missing_inputs_reported_before_output(tmp_path):
    with pytest.raises(ValueError, match='nfl-schedule.csv') as exc:
        builder.build(tmp_path/'missing', tmp_path/'result')
    assert 'cfb-2026.parquet' in str(exc.value)
    assert not (tmp_path/'result').exists()
    assert not list(tmp_path.glob('.*build*'))


def test_failed_verification_never_publishes(sources, tmp_path, monkeypatch):
    def fail(_):
        raise AssertionError('corrupted matrix')
    monkeypatch.setattr(builder, 'verify_tables', fail)
    output = tmp_path/'result'
    with pytest.raises(RuntimeError, match='corrupted matrix'):
        builder.build(sources, output)
    assert not output.exists()
    staging, = tmp_path.glob('.result.build-*')
    report = json.loads((staging/'build-report.json').read_text())
    assert report['status'] == 'failed' and report['active_step'] == 'verify_extracted'
    assert not (staging/'reset').exists()
    assert not (tmp_path/'.result.build.lock').exists()


def test_custom_nfl2023_path_and_concurrent_lock(sources, tmp_path):
    external = tmp_path/'custom.parquet'
    (sources/'play_by_play_2023.parquet').rename(external)
    lock = tmp_path/'.result.build.lock'
    lock.write_text('another builder')
    with pytest.raises(FileExistsError):
        builder.build(sources, tmp_path/'result', nfl2023=external)
    assert lock.read_text() == 'another builder'
    lock.unlink()
    assert builder.build(sources, tmp_path/'result', nfl2023=external).exists()


def test_source_mutation_blocks_publication(sources, tmp_path, monkeypatch):
    original = builder.verify_reset
    def mutate(**kwargs):
        original(**kwargs)
        with (sources/'nfl-schedule.csv').open('a') as stream:
            stream.write('\n')
    monkeypatch.setattr(builder, 'verify_reset', mutate)
    with pytest.raises(RuntimeError, match='Source input changed'):
        builder.build(sources, tmp_path/'result')
    assert not (tmp_path/'result').exists()
