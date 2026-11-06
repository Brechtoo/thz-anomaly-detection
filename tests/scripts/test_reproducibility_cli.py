"""CLI checks without raw data, models, training or external services."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
# Wiederverwendbare Prüf-CLIs werden unabhängig von lokalen Ergebnissen versioniert.
# Weiterhin echte Skripte prüfen; kein Fallback und kein Überspringen bei Verlust.
SCRIPTS = ROOT / 'scripts' / 'reproducibility'
OPTIONS = {
    'feature_probe.py': ['catalog', 'reference'],
    'verify_features.py': ['catalog', 'reference'],
    'verify_model_consistency.py': ['reference', 'rebuilt', 'model', 'predictions', 'reference-run-id'],
    'audit_catalog_leakage.py': ['raw-root', 'prior-check-dir'],
}


@pytest.mark.parametrize('script', OPTIONS)
def test_help_requires_no_historical_inputs(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), '--help'],
        cwd=tmp_path, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert '--output-dir' in result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('script,options', OPTIONS.items())
def test_existing_output_refused_before_reading_inputs(script, options, tmp_path):
    marker = tmp_path / 'existing.txt'
    marker.write_text('preserve me')
    arguments = [value for option in options for value in (f'--{option}', 'missing')]
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *arguments, '--output-dir', str(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 2
    assert 'Output directory must not exist' in result.stderr
    assert marker.read_text() == 'preserve me'
    assert list(tmp_path.iterdir()) == [marker]
