"""Integration checks for the source-tree migration and module CLIs."""
import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

@pytest.mark.parametrize('module', ['build_tables', 'verify_tables', 'fetch_raw_snapshot', 'audit_v3', 'build_artifacts', 'fetch_sources'])
def test_cli_help_without_data_or_network(module, tmp_path):
    result = subprocess.run([sys.executable, '-m', 'goalpost.transitions.' + module, '--help'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout
    assert not list(tmp_path.iterdir())


def test_every_transition_module_imports_without_data_io(tmp_path):
    # Plot modules must not read/write datasets when imported.
    import goalpost.transitions as package
    names = [p.stem for p in Path(package.__file__).parent.glob('*.py') if p.stem != '__init__']
    script = 'import importlib; ' + '; '.join('importlib.import_module(' + repr('goalpost.transitions.' + n) + ')' for n in names)
    env = {**os.environ, 'GOALPOST_TRANSITION_DATA': str(tmp_path/'absent')}
    result = subprocess.run([sys.executable, '-c', script], env=env, cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path/'absent').exists()


def test_artifact_override_is_outside_package(tmp_path):
    env = {**os.environ, 'GOALPOST_TRANSITION_DATA': str(tmp_path/'outputs')}
    result = subprocess.run([sys.executable, '-c',
        'from goalpost.transitions.paths import RESET; print(RESET)'], env=env,
        capture_output=True, text=True, check=True)
    assert result.stdout.strip() == str(tmp_path/'outputs/reset')


def test_current_and_strict_simulators_agree_when_no_grace_needed():
    from goalpost.transitions import run
    from goalpost.transitions.simulate_reset import run as strict
    from goalpost.simulator import run as public_run
    keys = np.array([[16,16,0,7,0],[16,16,0,3,0]])
    probs = np.zeros((2,9,2)); probs[0,:,0] = 1; probs[1,:,1] = 1
    timing = {(j,c): [1800.] for j in (0,1) for c in range(9)}
    old, _ = strict(keys, probs, timing, n=12)
    new, diagnostics, _, _ = run(keys, probs, timing, n=12)
    np.testing.assert_array_equal(old, new)
    assert diagnostics['completed'] == 12
    assert public_run is run
