import shutil

import numpy as np
import pandas as pd
import pytest

from src.data.split_manifest import sha256_file
from src.experiments import spatial_feature_inputs as inputs
from src.features.specimen_inputs import write_json


@pytest.fixture
def archive(tmp_path, monkeypatch):
    keys = inputs.KEYS
    train = pd.DataFrame([
        (f'{body:02}-{label}', f'body{body}', 'series', label, body // 3, float(body), float(label))
        for body in range(9) for label in range(4)
    ], columns=[*keys, 'x', 'y'])
    nested = tmp_path / 'nested'
    nested.mkdir()
    assignments_path = nested / 'inner_assignments.parquet'
    assignments = []
    for outer in range(3):
        allowed = train.loc[train.fold.ne(outer), keys]
        inner = allowed.specimen_id.map({body: i % 3 for i, body in enumerate(allowed.specimen_id.unique())})
        assignments.append(allowed.rename(columns={'fold': 'original_fold'}).assign(outer=outer, fold=inner))
    pd.concat(assignments).to_parquet(assignments_path, index=False)

    anchor, tables = tmp_path / 'anchor', tmp_path / 'tables'
    (anchor / 'features').mkdir(parents=True)
    tables.mkdir()
    for name, value in (('PROJECT_ROOT', tmp_path), ('ANCHOR', anchor), ('FEATURE_TABLES_ROOT', tables)):
        monkeypatch.setattr(inputs, name, value)
    # Cohort verification is covered by the original spatial-input tests.
    monkeypatch.setattr(inputs, '_frozen_cohort', lambda *_: train)
    monkeypatch.setattr(inputs, 'NestedFeatures', lambda df: df)
    cohort = anchor / 'cohort_folds.csv'
    train.to_csv(cohort, index=False)
    manual = anchor / 'features' / f'{inputs.EARLY}.npz'
    np.savez(manual, X=np.zeros((len(train), 52)))
    extraction = anchor / 'extraction_manifest.json'
    write_json(extraction, {'cohort_sha256': sha256_file(cohort),
                            'features_sha256': {manual.name: sha256_file(manual)}})
    anchor_plan = anchor / 'plan.json'
    write_json(anchor_plan, {})
    source_manifests = [extraction, anchor_plan]
    for block in ('tsfel', 'physical_v2'):
        table = tables / inputs.TABLE_FILES[block]
        train.to_parquet(table, index=False)
        manifest = table.with_suffix('.json')
        write_json(manifest, {'parquet_sha256': sha256_file(table)})
        source_manifests.append(manifest)

    plan = nested / 'plan.json'
    write_json(plan, {
        'protocol': 'nested_train_only_v1',
        'class_order': [0, 1, 2, 3],
        'features': {name: list(blocks) for name, blocks in inputs.FEATURES.items()},
        'source_sha256': {f'/old/{path.relative_to(tmp_path)}': sha256_file(path) for path in source_manifests},
        'inner_assignment_sha256': sha256_file(assignments_path),
    })
    return nested, train


def test_loader_pins_feature_tables_to_archived_manifests(archive):
    nested, train = archive
    loaded = inputs.load_spatial_feature_inputs(nested)
    pd.testing.assert_frame_equal(loaded.train, train.assign(split='train'))
    assert not loaded.assignments.duplicated(['outer', 'instance_id']).any()
    table = inputs.FEATURE_TABLES_ROOT / inputs.TABLE_FILES['tsfel']
    assert loaded.hashes[str(table)] == sha256_file(table)

    train.iloc[:-1].to_parquet(table, index=False)
    with pytest.raises(ValueError, match='Checksum mismatch'):
        inputs.load_spatial_feature_inputs(nested)


def test_loader_detects_positional_manual_row_drift(archive):
    nested, train = archive
    train.iloc[::-1].to_csv(inputs.ANCHOR / 'cohort_folds.csv', index=False)
    with pytest.raises(ValueError, match='Manual feature rows'):
        inputs.load_spatial_feature_inputs(nested)


def test_reference_checks_logged_copy_and_matches_native_prediction(archive):
    nested, train = archive
    checks = [nested / 'plan.json', nested / 'inner_assignments.parquet']
    for fold, held in train.groupby('fold'):
        directory = nested / 'outer' / 'manual' / f'fold_{fold}'
        directory.mkdir(parents=True)
        stored = directory / 'predictions.parquet'
        held[inputs.KEYS].assign(prediction_regular=held.label).to_parquet(stored, index=False)
        provenance = directory / 'provenance.json'
        write_json(provenance, {'sha256': sha256_file(stored)})
        checks.extend([stored, provenance])

    post = nested.parent / 'post'
    uid = 'a' * 32
    logged = post / 'mlartifacts' / uid / 'artifacts'
    logged.mkdir(parents=True)
    write_json(post / 'run.json', {'run_id': uid})
    write_json(post / 'plan.json', {
        'protocol': 'spatial_nested_train_oof_v1', 'baselines': ['manual'],
        'inputs_sha256': {f'/old/{path.relative_to(nested.parent)}': sha256_file(path) for path in checks},
    })
    shutil.copyfile(post / 'plan.json', logged / 'plan.json')
    original = train.assign(baseline='manual', strategy='baseline', prediction=train.label,
                            baseline_prediction=train.label)
    rescue = original.assign(strategy='rescue', prediction=train.label.replace(3, 2))
    paired = pd.concat([original, rescue]).sample(frac=1, random_state=42)
    path = post / 'predictions.parquet'
    paired.to_parquet(path, index=False)
    shutil.copyfile(path, logged / path.name)

    result, hashes = inputs.load_spatial_feature_reference(nested, post, train, ['manual'])
    assert result.groupby('arm').size().to_dict() == {'prior_native': len(train), 'prior_rescue': len(train)}
    assert result.query("arm == 'prior_native'").prediction.tolist() == train.label.tolist()
    assert hashes[str(path)] == sha256_file(logged / path.name)

    paired.assign(baseline_prediction=0).to_parquet(path, index=False)
    with pytest.raises(ValueError, match='Checksum mismatch'):
        inputs.load_spatial_feature_reference(nested, post, train, ['manual'])
    shutil.copyfile(path, logged / path.name)
    with pytest.raises(ValueError, match='do not match'):
        inputs.load_spatial_feature_reference(nested, post, train, ['manual'])
