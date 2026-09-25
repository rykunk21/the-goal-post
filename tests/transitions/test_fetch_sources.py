"""No network in tests: actual CSV/parquet payloads exercise download validation."""
import io
import json
from urllib.error import HTTPError

import pandas as pd
import pytest
from goalpost.transitions import fetch_sources as downloader


class Response(io.BytesIO):
    def __init__(self, payload, length=None):
        super().__init__(payload)
        self.headers = {'Content-Length': str(len(payload) if length is None else length), 'ETag': 'test'}


@pytest.fixture
def payloads():
    result = {}
    for name, url, columns in downloader.source_plan():
        frame = pd.DataFrame([{key: 1 for key in columns}])
        if name.endswith('.csv'):
            payload = frame.to_csv(index=False).encode()
        else:
            stream = io.BytesIO()
            frame.to_parquet(stream, index=False)
            payload = stream.getvalue()
        result[url] = payload
    return result


def test_all_required_sources_download_and_reuse(payloads, monkeypatch, tmp_path):
    calls = []
    def open_url(request, timeout):
        calls.append(request.full_url)
        return Response(payloads[request.full_url])
    monkeypatch.setattr(downloader, 'urlopen', open_url)
    monkeypatch.setenv('GOALPOST_TRANSITION_DATA', str(tmp_path/'ignored'))
    monkeypatch.chdir(tmp_path)
    output = downloader.fetch()
    assert output == tmp_path/'artifacts/sources'
    assert len(calls) == 13
    report = json.loads((output/'download-manifest.json').read_text())
    assert report['status'] == 'complete' and len(report['files']) == 13
    for name, _, _ in downloader.source_plan():
        assert downloader.sha256(output/name) == report['files'][name]['sha256']
    downloader.fetch()
    assert len(calls) == 13
    assert not (tmp_path/'ignored').exists()


def test_failed_download_resumes_without_empty_substitution(payloads, monkeypatch, tmp_path):
    calls = []
    fail_url = downloader.source_plan()[2][1]
    def open_url(request, timeout):
        calls.append(request.full_url)
        if request.full_url == fail_url:
            raise HTTPError(fail_url, 404, 'Missing season', {}, None)
        return Response(payloads[request.full_url])
    monkeypatch.setattr(downloader, 'urlopen', open_url)
    with pytest.raises(RuntimeError, match='404'):
        downloader.fetch(tmp_path)
    assert not (tmp_path/'cfb-2023.parquet').exists()
    assert not list(tmp_path.glob('*.part'))
    assert not (tmp_path/'.download.lock').exists()
    report = json.loads((tmp_path/'download-manifest.json').read_text())
    assert report['status'] == 'failed' and len(report['files']) == 2
    monkeypatch.setattr(downloader, 'urlopen', lambda request, timeout: Response(payloads[request.full_url]))
    downloader.fetch(tmp_path)
    assert json.loads((tmp_path/'download-manifest.json').read_text())['status'] == 'complete'


@pytest.mark.parametrize('payload,length,match', [
    (b'not,a,schedule\n1,2,3\n', None, 'missing required columns'),
    (b'abc', 100, 'incomplete download'),
])
def test_invalid_payload_not_published(monkeypatch, tmp_path, payload, length, match):
    monkeypatch.setattr(downloader, 'urlopen', lambda *a, **kw: Response(payload, length))
    with pytest.raises(RuntimeError, match=match):
        downloader.fetch(tmp_path)
    assert not (tmp_path/'nfl-schedule.csv').exists()


def test_untracked_file_not_trusted(monkeypatch, tmp_path):
    (tmp_path/'nfl-schedule.csv').write_text('unknown')
    monkeypatch.setattr(downloader, 'urlopen', lambda *a, **kw: pytest.fail('Must not download'))
    with pytest.raises(ValueError, match='Unverified'):
        downloader.fetch(tmp_path)


def test_changed_cached_file_rejected(payloads, monkeypatch, tmp_path):
    monkeypatch.setattr(downloader, 'urlopen', lambda request, timeout: Response(payloads[request.full_url]))
    downloader.fetch(tmp_path)
    (tmp_path/'nfl-schedule.csv').write_text('changed')
    with pytest.raises(ValueError, match='changed local file'):
        downloader.fetch(tmp_path)


def test_lock_is_not_removed_by_competing_invocation(tmp_path):
    (tmp_path/'.download.lock').write_text('another process')
    with pytest.raises(FileExistsError):
        downloader.fetch(tmp_path)
    assert (tmp_path/'.download.lock').read_text() == 'another process'
