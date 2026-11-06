"""Download the public historical files consumed by build_artifacts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow.parquet as pq

from .build_artifacts import sha256
from .build_tables import CFB_COLUMNS, CUTOFF

NFL_COLUMNS = {
    'game_id', 'play_id', 'home_team', 'away_team', 'home_score', 'away_score',
    'posteam', 'posteam_score_post', 'defteam_score_post', 'total_home_score',
    'total_away_score', 'down', 'play_type', 'ydstogo', 'yardline_100',
    'qtr', 'half_seconds_remaining', 'week', 'game_date',
}
NFL_SCHEDULE_COLUMNS = {
    'game_id', 'season', 'week', 'game_type', 'gameday', 'home_team',
    'away_team', 'home_score', 'away_score',
}
COLLEGE_SCHEDULE_COLUMNS = {
    'game_id', 'season', 'week', 'season_type', 'start_date', 'completed',
    'home_team', 'away_team', 'home_id', 'away_id', 'home_division',
    'away_division', 'home_points', 'away_points',
}


def source_plan():
    """Exact source families, filenames and seasons required by the extractor."""
    result = [('nfl-schedule.csv',
               'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv',
               NFL_SCHEDULE_COLUMNS)]
    for year in range(2023, 2027):
        result.extend([
            ('play_by_play_2023.parquet' if year == 2023 else f'nfl-{year}.parquet',
             f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet',
             NFL_COLUMNS),
            (f'cfb-{year}.parquet',
             f'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfbfastR_cfb_pbp/play_by_play_{year}.parquet',
             set(CFB_COLUMNS)),
            (f'cfb_schedules-{year}.parquet',
             f'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfb_schedules/cfb_schedules_{year}.parquet',
             COLLEGE_SCHEDULE_COLUMNS),
        ])
    return result


def inspect(path, name, required):
    if name.endswith('.csv'):
        frame = pd.read_csv(path)
        columns, rows = set(frame.columns), len(frame)
    else:
        parquet = pq.ParquetFile(path)
        columns, rows = set(parquet.schema_arrow.names), parquet.metadata.num_rows
    missing = required - columns
    if missing:
        raise ValueError(f'{name}: missing required columns: {sorted(missing)}')
    if rows == 0:
        raise ValueError(f'{name}: upstream file is empty')
    return dict(rows=rows, columns=sorted(columns))


def write_manifest(path, report):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def fetch(output='artifacts/sources', timeout=120):
    """Download once; reuse only manifest-bound files with matching hashes.

    To refresh mutable upstream releases, choose a new output directory.
    A missing upstream season is an error, never an invented empty parquet.
    """
    if timeout <= 0:
        raise ValueError('Timeout must be positive')
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'download-manifest.json'
    lock = output/'.download.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    try:
        report = json.loads(manifest.read_text()) if manifest.exists() else dict(version=1, files={})
        if report.get('version') != 1 or not isinstance(report.get('files'), dict):
            raise ValueError('Unsupported download manifest')
        # Check the entire cache before changing any of it.
        for name, url, _ in source_plan():
            path = output/name
            receipt = report['files'].get(name)
            if path.exists() and (not receipt or receipt.get('url') != url or receipt.get('sha256') != sha256(path)):
                raise ValueError(f'Unverified or changed local file: {path}. Choose a new output directory.')
        report.update(status='downloading', extraction_cutoff=CUTOFF,
                      note='Mutable upstream downloads; not a byte-identical reproduction of the original frozen sources.')
        write_manifest(manifest, report)
        try:
            for name, url, required in source_plan():
                path = output/name
                if path.exists():
                    inspect(path, name, required)
                    print(f'Cached and verified: {name}', flush=True)
                    continue
                print(f'Downloading: {name}', flush=True)
                report['active_file'] = name
                write_manifest(manifest, report)
                fd, temporary_name = tempfile.mkstemp(prefix=f'.{name}.', suffix='.part', dir=output)
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(fd, 'wb') as destination:
                        request = Request(url, headers={'User-Agent': 'goalpost-source-builder/1.0'})
                        with urlopen(request, timeout=timeout) as response:
                            expected = response.headers.get('Content-Length')
                            headers = {key: response.headers.get(key) for key in ('ETag', 'Last-Modified')}
                            while chunk := response.read(1024 * 1024):
                                destination.write(chunk)
                    size = temporary.stat().st_size
                    if expected is not None and size != int(expected):
                        raise ValueError(f'{name}: incomplete download ({size}/{expected} bytes)')
                    metadata = inspect(temporary, name, required)
                    receipt = dict(url=url, sha256=sha256(temporary), bytes=size,
                                   retrieved_at=datetime.now(timezone.utc).isoformat(),
                                   response_headers=headers, **metadata)
                    # Publish receipt first: a crash before the file rename is safely resumable.
                    report['files'][name] = receipt
                    write_manifest(manifest, report)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            report.update(status='complete', active_file=None,
                          completed_at=datetime.now(timezone.utc).isoformat())
            report.pop('error', None)
            write_manifest(manifest, report)
        except Exception as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            write_manifest(manifest, report)
            raise RuntimeError(f'Source download failed: {exc}. See {manifest}; rerun to resume verified files.') from exc
    finally:
        lock.unlink()
    print(f'Sources ready: {output}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/sources'))
    parser.add_argument('--timeout', type=float, default=120, help='Network read timeout in seconds')
    args = parser.parse_args()
    try:
        fetch(args.output, args.timeout)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
