"""Build and verify all weekly input artifacts without environment configuration."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from .build_tables import export, CUTOFF
from .verify_tables import verify as verify_tables
from .build_reset import rebuild
from .verify_reset import verify as verify_reset


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def build(sources, output='artifacts/transitions', nfl2023=None, raw_summaries=None):
    """Publish a new artifact root only after both independent verifiers pass.

    Failed work and its log remain in a sibling staging directory for diagnosis.
    Existing output roots are never reused. No source acquisition occurs here.
    """
    if not __debug__:
        raise ValueError('Verification requires assertions: do not run Python with -O.')
    sources = Path(sources).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    nfl2023 = Path(nfl2023).expanduser().resolve() if nfl2023 else sources/'play_by_play_2023.parquet'
    raw_summaries = Path(raw_summaries).expanduser().resolve() if raw_summaries else None
    required = [sources/'nfl-schedule.csv', nfl2023]
    required += [sources/f'nfl-{year}.parquet' for year in range(2024, 2027)]
    required += [sources/f'{prefix}-{year}.parquet'
                 for prefix in ('cfb_schedules', 'cfb') for year in range(2023, 2027)]
    missing = [str(path) for path in required if not path.is_file()]
    if raw_summaries is not None and not raw_summaries.is_dir():
        missing.append(str(raw_summaries) + ' (directory)')
    if missing:
        raise ValueError('Missing source inputs:\n' + '\n'.join(missing))
    if output.exists():
        raise ValueError(f'Refusing to overwrite existing artifact root: {output}')
    if output == sources or output in sources.parents:
        raise ValueError('Output cannot contain the source directory.')
    optional = sources/'espn_cfb_pbp-2026.parquet'
    inputs = required + ([optional] if optional.is_file() else [])
    if raw_summaries:
        inputs += sorted(p for p in raw_summaries.rglob('*') if p.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent/f'.{output.name}.build.lock'
    # Exclusive creation prevents concurrent invocations from publishing this target.
    with lock.open('x') as handle:
        handle.write(str(output) + '\n')
    staging = None
    try:
        if output.exists():
            raise ValueError(f'Output appeared while acquiring build lock: {output}')
        staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.build-', dir=output.parent))
        report = dict(status='building', extraction_cutoff=CUTOFF,
                      output=str(output), staging=str(staging), steps=[],
                      started_at=datetime.now(timezone.utc).isoformat(),
                      source_hashes={str(p): sha256(p) for p in inputs})
        save(staging/'build-report.json', report)
        steps = [
            ('extract', lambda: export(sources, nfl2023, staging/'extracted', raw_summaries=raw_summaries)),
            ('verify_extracted', lambda: verify_tables(staging/'extracted')),
            ('reset_and_impute', lambda: rebuild(root=staging/'reset', source=staging/'extracted')),
            ('verify_reset', lambda: verify_reset(root=staging/'reset', source=staging/'extracted')),
        ]
        try:
            with (staging/'build.log').open('w') as log:
                for name, action in steps:
                    print(f'Building: {name}', flush=True)
                    report['active_step'] = name
                    save(staging/'build-report.json', report)
                    with redirect_stdout(log), redirect_stderr(log):
                        action()
                    log.flush()
                    report['steps'].append(name)
                for verification in (staging/'extracted/verification.json', staging/'reset/verification.json'):
                    if json.loads(verification.read_text()).get('passed') is not True:
                        raise ValueError(f'Verification did not pass: {verification}')
                if any(sha256(Path(p)) != digest for p, digest in report['source_hashes'].items()):
                    raise ValueError('Source input changed during build; refusing to publish.')
            # Rebase the provenance path for the atomic directory publication.
            summary_path = staging/'reset/data/summary.json'
            summary = json.loads(summary_path.read_text())
            summary['raw_source'] = str(output/'extracted')
            save(summary_path, summary)
            report.update(status='complete', active_step=None,
                          completed_at=datetime.now(timezone.utc).isoformat(),
                          weekly_data=str(output/'reset/data'))
            report['artifact_hashes'] = {
                str(p.relative_to(staging)): sha256(p)
                for folder in ('extracted', 'reset')
                for p in sorted((staging/folder).rglob('*')) if p.is_file()}
            save(staging/'build-report.json', report)
            if output.exists():
                raise ValueError(f'Output appeared during build: {output}')
            staging.rename(output)
        except Exception as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            save(staging/'build-report.json', report)
            raise RuntimeError(f'Build failed; unpublished diagnostics retained at {staging}: {exc}') from exc
    finally:
        lock.unlink()
    print(f'Verified weekly artifacts: {output / "reset/data"}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True, help='Directory of historical NFL and college source files')
    parser.add_argument('--output', type=Path, default=Path('artifacts/transitions'), help='New artifact root (default: artifacts/transitions; ignores environment overrides)')
    parser.add_argument('--nfl2023', type=Path, help='Defaults to SOURCES/play_by_play_2023.parquet')
    parser.add_argument('--raw-summaries', type=Path, help='Optional saved ESPN college summary directory')
    args = parser.parse_args()
    try:
        build(args.sources, args.output, args.nfl2023, args.raw_summaries)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
