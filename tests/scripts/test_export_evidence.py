"""Publication copies must be explicit, traceable and leave originals intact."""
import hashlib
import json

import pytest

from scripts.maintenance.export_evidence import export_evidence


def test_copy_and_csv_projection_preserve_source_and_record_hashes(tmp_path):
    source = tmp_path / 'run.csv'
    content = b'model,accuracy,file_path\nhgb,0.9,/Users/local/raw.txt\n'
    source.write_bytes(content)
    out = tmp_path / 'evidence'
    export_evidence(tmp_path, [dict(source='run.csv', destination='summary.csv', columns=['model', 'accuracy'])], out)
    assert source.read_bytes() == content
    assert (out / 'summary.csv').read_text() == 'model,accuracy\nhgb,0.9\n'
    item = json.loads((out / 'manifest.json').read_text())['files'][0]
    assert item['source_sha256'] == hashlib.sha256(content).hexdigest()
    assert item['exported_sha256'] == hashlib.sha256((out / 'summary.csv').read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        export_evidence(tmp_path, [], out)


@pytest.mark.parametrize('specification', [
    [dict(source='../outside.csv', destination='result.csv')],
    [dict(source='run.csv', destination='../result.csv')],
    [dict(source='run.csv', destination='manifest.json')],
    [dict(source='run.csv', destination='a.csv')] * 2,
    [dict(source='private.csv', destination='a.csv')],
])
def test_invalid_or_private_inputs_do_not_create_output(tmp_path, specification):
    (tmp_path / 'run.csv').write_text('value\n1\n')
    (tmp_path / 'private.csv').write_text('file_path\n/Users/local/raw.txt\n')
    destination = tmp_path / 'out'
    with pytest.raises((ValueError, FileNotFoundError)):
        export_evidence(tmp_path, specification, destination)
    assert not destination.exists()
