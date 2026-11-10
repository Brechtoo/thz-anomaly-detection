"""Synthetic archive evidence verifies reuse, tamper checks and train-only medians."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.dataset import EXCLUDE_COLUMNS
from src.data.specimen_splits import load_membership, write_specimen_splits
from src.data.split_manifest import sha256_file
from src.features.specimen_manual import (
    prepare_manual_features,
    validate_manual_schema,
    verify_existing_features,
)
from src.paths import PROJECT_ROOT


@pytest.fixture
def inputs(tmp_path):
    rows, mapping = [], []
    for g in range(35):
        for label in range(4):
            iid = f'id_{g:02d}_{label}'
            signal = tmp_path / f'{iid}.txt'
            signal.write_text(f'{g} {label}\n')
            rows.append([label, str(signal), g, label, f's{g % 3}', iid])
            mapping.append([iid, f'pk_{g:02d}'])
    catalog = pd.DataFrame(rows, columns=EXCLUDE_COLUMNS)
    catalog_path = tmp_path / 'catalog.csv'
    catalog.to_csv(catalog_path, index=False)
    mapping_path = tmp_path / 'mapping.csv'
    pd.DataFrame(mapping, columns=['instance_id', 'specimen_id']).to_csv(mapping_path, index=False)
    split_dir = tmp_path / 'splits'
    membership = write_specimen_splits(catalog_path, mapping_path, split_dir, candidates=16)
    columns = [f'f{i}' for i in range(52)]
    table = catalog.copy()
    for i, name in enumerate(columns):
        table[name] = np.arange(len(table), dtype=float) + i
    training_ids = membership.loc[membership.split.eq('train'), 'instance_id']
    heldout = ~table.instance_id.isin(training_ids)
    table.loc[heldout, 'f0'] = 1e9
    table.loc[table.index[table.instance_id.isin(training_ids)][0], 'f0'] = np.nan
    table_path = tmp_path / 'manual.csv'
    table.to_csv(table_path, index=False)
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    table.to_csv(evidence / 'feature_table.csv', index=False)
    pd.DataFrame({'instance_id': table.instance_id, 'file_path': table.file_path,
                  'sha256': [sha256_file(p) for p in table.file_path]}).to_csv(evidence / 'raw_inputs.csv', index=False)
    sources = ['loading.py', 'extract_all_features.py', 'extract_baseline_features.py',
               'extract_extension_block1_features.py']
    manifest = {'output_csv_sha256': sha256_file(evidence / 'feature_table.csv'),
                'raw_inputs_sha256': sha256_file(evidence / 'raw_inputs.csv'),
                'input_csv_sha256': sha256_file(catalog_path), 'feature_columns': columns,
                'source_sha256': {'src/features/' + p: sha256_file(PROJECT_ROOT / 'src/features' / p) for p in sources}}
    (evidence / 'manifest.json').write_text(json.dumps(manifest))
    schema = tmp_path / 'schema.json'
    schema.write_text(json.dumps({'features': columns, 'source_table_sha256': sha256_file(table_path)}))
    return table_path, catalog_path, split_dir, schema, evidence, columns


def test_reuses_data_and_fits_imputation_on_training_only(inputs, tmp_path):
    table, catalog, splits, schema, evidence, columns = inputs
    before = [p.read_bytes() for p in [table, catalog, splits / 'manifest.json']]
    output = tmp_path / 'prepared'
    prepare_manual_features(table, catalog, splits, schema, evidence, output)
    membership = load_membership(splits)
    original = pd.read_csv(table).set_index('instance_id')
    train = original.loc[membership.loc[membership.split.eq('train'), 'instance_id'], columns]
    imputer = joblib.load(output / 'imputer.joblib')
    np.testing.assert_allclose(imputer.statistics_, train.median().to_numpy())
    assert imputer.statistics_[0] < 140  # huge held-out values must not influence the median
    result = pd.read_csv(output / 'features.csv')
    assert result.instance_id.tolist() == membership.instance_id.tolist()
    assert len(result) == 140 and np.isfinite(result[columns]).all().all()
    assert set(result) == {'instance_id', 'label', *columns}
    assert [p.read_bytes() for p in [table, catalog, splits / 'manifest.json']] == before
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['feature_set'] == 'manual52' and manifest['training_only']
    assert json.loads((output / 'verification.json').read_text())['new_extraction'] is False
    with pytest.raises(FileExistsError):
        prepare_manual_features(table, catalog, splits, schema, evidence, output)


@pytest.mark.parametrize('problem', ['raw', 'values', 'missing'])
def test_rejects_invalid_existing_evidence(inputs, problem):
    table, catalog, _, _, evidence, columns = inputs
    data = pd.read_csv(table)
    if problem == 'raw':
        from pathlib import Path
        Path(data.file_path.iloc[0]).write_text('changed measurement')
    elif problem == 'values':
        data.loc[0, 'f1'] += 1
        data.to_csv(table, index=False)
    else:
        data.iloc[:-1].to_csv(table, index=False)
    with pytest.raises(ValueError):
        verify_existing_features(table, catalog, evidence, columns)


@pytest.mark.parametrize('metadata', ['x', 'y', 'label', 'specimen_id', 'training_eligible', 'split'])
def test_schema_never_allows_metadata(metadata):
    columns = [f'f{i}' for i in range(51)] + [metadata]
    with pytest.raises(ValueError, match='non-metadata'):
        validate_manual_schema(columns)


def test_source_audit_accepts_only_formatting_changes(tmp_path, monkeypatch):
    import hashlib

    from src.features import specimen_manual as module
    archived = b'def feature(x):\n    return x + 1\n'
    names = ['loading.py', 'extract_all_features.py', 'extract_baseline_features.py',
             'extract_extension_block1_features.py']
    manifest = {'source_sha256': {}, 'git_state': {'git_commit': 'synthetic', 'tracked_diff': ''}}
    for name in names:
        target = tmp_path / 'src/features' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('# Formatting only\ndef feature(x):\n\n    return x+1\n')
        manifest['source_sha256']['src/features/' + name] = hashlib.sha256(archived).hexdigest()
    monkeypatch.setattr(module, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *args, **kwargs: archived)
    assert module.verify_extractor_sources(manifest)['comparison'] == 'AST_identical'
    (tmp_path / 'src/features/loading.py').write_text('def feature(x):\n    return x + 2\n')
    with pytest.raises(ValueError, match='calculations changed'):
        module.verify_extractor_sources(manifest)
