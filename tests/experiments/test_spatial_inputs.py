import numpy as np
import pandas as pd
import pytest

from src.data.split_manifest import sha256_file
from src.evaluation.svm_nested_selection import (
    SCORE_COLUMNS,
    fit_calibrator,
    predict_probabilities,
)
from src.experiments import spatial_inputs
from src.features.specimen_inputs import read_json, write_json


@pytest.fixture
def nested_archive(tmp_path, monkeypatch):
    paths = {name: tmp_path / name.lower() for name in ('ANCHOR', 'FOLDS', 'SPLITS')}
    for name, path in paths.items():
        path.mkdir()
        monkeypatch.setattr(spatial_inputs, name, path)
    monkeypatch.setattr(spatial_inputs, 'PROJECT_ROOT', tmp_path)
    source = tmp_path / 'nested'
    source.mkdir()
    keys = spatial_inputs.KEYS
    probabilities = spatial_inputs.PROBABILITIES

    rows = []
    for body in range(9):
        for label in range(4):
            rows.append({
                'instance_id': f'messreihe_1_x{body}_y{label}',
                'specimen_id': f'specimen_{body}',
                'source_dataset': 'messreihe_1',
                'label': label,
                'fold': body // 3,
                'file_path': f'/not/read/2026-01-01T01-02-03.456-S-[1]-'
                f'[{body + .1},{label + .2},0]-[1,0,0,0]-delta[0]-avg20.txt',
            })
    cohort = pd.DataFrame(rows).sort_values('instance_id').reset_index(drop=True)
    anchor = paths['ANCHOR'] / 'cohort_folds.csv'
    cohort.to_csv(anchor, index=False)
    train = paths['SPLITS'] / 'train.csv'
    cohort[['instance_id', 'label']].to_csv(train, index=False)
    folds = paths['FOLDS'] / 'specimen_folds.csv'
    cohort[['specimen_id', 'fold']].drop_duplicates().to_csv(folds, index=False)
    manifest = paths['FOLDS'] / 'manifest.json'
    write_json(manifest, {
        'train_split_sha256': sha256_file(train), 'source_sha256': sha256_file(anchor),
        'files': {folds.name: sha256_file(folds)},
        'points_per_fold': {str(fold): 12 for fold in range(3)},
        'bodies_per_fold': {str(fold): 3 for fold in range(3)},
    })
    predictions = cohort[keys].assign(prediction=cohort.label)
    predictions[SCORE_COLUMNS] = np.eye(4)[cohort.label]
    assignments = []
    for outer in range(3):
        inner = predictions.loc[predictions.fold.ne(outer)].copy()
        groups = sorted(inner.specimen_id.unique())
        by_group = {group: i % 3 for i, group in enumerate(groups)}
        inner = inner.assign(original_fold=inner.fold, outer=outer, fold=inner.specimen_id.map(by_group))
        assignments.append(inner[[*keys, 'original_fold', 'outer']])
        work = source / 'search' / 'manual' / f'outer_{outer}'
        for inner_fold, part in inner.groupby('fold'):
            directory = work / f'inner_{inner_fold}' / 'fixed'
            directory.mkdir(parents=True)
            path = directory / 'predictions.parquet'
            part[[*keys, 'prediction', *SCORE_COLUMNS]].to_parquet(path, index=False)
            write_json(directory / 'fit.json', {'sha256': sha256_file(path)})
        write_json(work / 'selection.json', {'regular': {'candidate': 'fixed', 'calibration': 'multinomial'}})
        allowed = inner.sort_values(['fold', 'instance_id'])
        calibrator = fit_calibrator(allowed[SCORE_COLUMNS].to_numpy(), allowed.label.to_numpy())
        held = predictions.loc[predictions.fold.eq(outer)].copy()
        native = work / 'outer_fits' / 'fixed' / 'predictions.parquet'
        native.parent.mkdir(parents=True)
        held.to_parquet(native, index=False)
        held[probabilities] = predict_probabilities(calibrator, held[SCORE_COLUMNS].to_numpy())
        held = held.rename(columns={col: f'regular_{col}' for col in [*probabilities, *SCORE_COLUMNS]})
        directory = source / 'outer' / 'manual' / f'fold_{outer}'
        directory.mkdir(parents=True)
        output = directory / 'predictions.parquet'
        held.rename(columns={'prediction': 'prediction_regular'}).to_parquet(output, index=False)
        write_json(directory / 'provenance.json', {
            'sha256': sha256_file(output),
            'calibration': {'regular': {'method': 'multinomial', 'inputs': len(inner),
                                        'fit_times': {'sha256': sha256_file(native)}}},
        })
    inner_path = source / 'inner_assignments.parquet'
    pd.concat(assignments, ignore_index=True).to_parquet(inner_path, index=False)
    write_json(source / 'plan.json', {
        'protocol': 'nested_train_only_v1', 'class_order': [0, 1, 2, 3],
        'features': {'manual': ['manual']}, 'candidates': [{'candidate': 'fixed'}],
        'source_sha256': {f'/original/{path.relative_to(tmp_path)}': sha256_file(path) for path in (manifest, train)},
        'inner_assignment_sha256': sha256_file(inner_path),
    })
    return source


def test_training_only_nested_inputs_preserve_native_order_and_recorded_coordinates(nested_archive):
    folds, digests = spatial_inputs.load_spatial_input(nested_archive, 'manual')
    assert [fold.fold for fold in folds] == [0, 1, 2]
    outer = pd.concat([fold.outer for fold in folds], ignore_index=True)
    assert len(outer) == 36 and outer.instance_id.is_unique
    assert outer.prediction.equals(outer.label)
    np.testing.assert_allclose(outer.y, outer.label + .2)
    for fold in folds:
        assert len(fold.inner) == 24
        assert set(fold.inner.specimen_id).isdisjoint(fold.outer.specimen_id)
        assert fold.inner.groupby('specimen_id').fold.nunique().eq(1).all()
    assert all('validation.csv' not in path and 'test.csv' not in path for path in digests)


def test_stale_inner_prediction_digest_fails_closed(nested_archive):
    directory = nested_archive / 'search/manual/outer_0/inner_0/fixed'
    data = read_json(directory / 'fit.json')
    write_json(directory / 'fit.json', {**data, 'sha256': '0' * 64})
    with pytest.raises(ValueError, match='Checksum mismatch'):
        spatial_inputs.load_spatial_input(nested_archive, 'manual')


def test_inner_assignment_cannot_substitute_an_outer_measurement(nested_archive):
    path = nested_archive / 'inner_assignments.parquet'
    assignments = pd.read_parquet(path)
    assignments.loc[0, 'instance_id'] = 'messreihe_1_x0_y0'
    assignments.to_parquet(path, index=False)
    plan_path = nested_archive / 'plan.json'
    plan = read_json(plan_path)
    write_json(plan_path, {**plan, 'inner_assignment_sha256': sha256_file(path)})
    with pytest.raises(ValueError, match='Prediction ID coverage differs'):
        spatial_inputs.load_spatial_input(nested_archive, 'manual')


def test_outer_probabilities_must_match_calibration_on_allowed_bodies(nested_archive):
    directory = nested_archive / 'outer/manual/fold_0'
    path = directory / 'predictions.parquet'
    frame = pd.read_parquet(path)
    frame[[f'regular_p_{label}' for label in range(4)]] = .25
    frame.to_parquet(path, index=False)
    provenance_path = directory / 'provenance.json'
    provenance = read_json(provenance_path)
    write_json(provenance_path, {**provenance, 'sha256': sha256_file(path)})
    with pytest.raises(ValueError, match='disagree with training-only calibration'):
        spatial_inputs.load_spatial_input(nested_archive, 'manual')
