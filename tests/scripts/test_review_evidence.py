"""Public review evidence must work without ignored runs or measurement data."""
import hashlib
import json
from pathlib import Path


def test_selected_evidence_matches_export_hashes_without_local_runs():
    root = Path(__file__).resolve().parents[2]
    folder = root / 'docs/evidence'
    manifest = json.loads((folder / 'manifest.json').read_text())
    selection = json.loads((root / 'configs/review_evidence.json').read_text())
    assert {(v['source'], v['destination']) for v in manifest['files']} == {(v['source'], v['destination']) for v in selection}
    for item in manifest['files']:
        path = folder / item['destination']
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['exported_sha256']
        assert path.stat().st_size == item['bytes']
        if not item['columns']:
            assert item['source_sha256'] == item['exported_sha256']
