import numpy as np
import pandas as pd
import pytest

from src.analysis.tabular_nested_report import bootstrap
from src.data.split_manifest import sha256_file
from src.experiments.svm_nested_hpo import KEYS, SCORES, fit_checkpoint
from src.experiments.tabular_nested_hpo import (
    fit_one,
    freeze,
    freeze_plan,
    verified_assignments,
)
from src.experiments.tabular_nested_space import MODELS, grid


def cohort():
    return pd.DataFrame([
        dict(zip(KEYS, (f'{fold}-{body}-{label}', f'{fold}-{body}', 'train_source', label, fold), strict=True))
        for fold in range(3) for body in range(3) for label in range(4)
    ])


def test_inner_assignments_reject_a_specimen_split(tmp_path):
    train = cohort()
    parts = []
    for outer in range(3):
        eligible = train.loc[train.fold.ne(outer)].rename(columns={'fold': 'original_fold'})
        parts.append(eligible.assign(outer=outer, fold=eligible.specimen_id.str[-1].astype(int)))
    assignments = pd.concat(parts, ignore_index=True)
    path = tmp_path / 'inner.parquet'
    assignments.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(verified_assignments(train, path, sha256_file(path)), assignments)
    assignments.loc[0, 'fold'] = 1
    assignments.to_parquet(path, index=False)
    with pytest.raises(ValueError, match='Invalid frozen inner split'):
        verified_assignments(train, path, sha256_file(path))


def test_grid_and_verified_fit_resume(tmp_path, monkeypatch):
    for model in MODELS:
        candidates = grid(model)
        assert len({row['candidate'] for row in candidates}) == 16
        assert candidates[0]['candidate'] == 'reference'
    train = cohort()
    fit, held = train.loc[train.fold.ne(0)], train.loc[train.fold.eq(0)]
    X, Z = np.eye(4)[fit.label], np.eye(4)[held.label]
    candidate = grid('logistic_regression')[13]
    checkpoint = tmp_path / 'fit'
    predictions = fit_one(checkpoint, 'logistic_regression', candidate, fit, held, X, Z)
    assert predictions.prediction.tolist() == held.label.tolist()
    monkeypatch.setattr('src.experiments.tabular_nested_hpo.create_model', None)
    pd.testing.assert_frame_equal(
        fit_one(checkpoint, 'logistic_regression', candidate, fit, held, X, Z), predictions,
    )


def test_paired_bootstrap_retains_the_direction_of_critical_errors():
    train = cohort()
    old = train.assign(prediction_regular=train.label)
    new = old.assign(prediction_regular=np.where(train.label.eq(1), 3, train.label))
    assert bootstrap(new, old, 'regular', draws=40) == [50., 50.]


def test_single_precision_checkpoint_is_reused_as_double_for_calibration(tmp_path, monkeypatch):
    held = cohort()
    directory = tmp_path / 'fit'
    scores = np.random.default_rng(42).normal(size=(len(held), 4)).astype('float32')
    fit_checkpoint(directory, held, held.label.to_numpy(), scores, {})
    digest = sha256_file(directory / 'predictions.parquet')
    monkeypatch.setattr('src.experiments.tabular_nested_hpo.create_model', None)
    frame = fit_one(directory, 'xgboost', {}, None, held, None, None)
    assert frame[SCORES].to_numpy().dtype == np.float64
    np.testing.assert_array_equal(frame[SCORES].to_numpy(), scores)
    assert sha256_file(directory / 'predictions.parquet') == digest


def test_precision_revision_preserves_the_original_plan_and_pins_the_updated_source(tmp_path, monkeypatch):
    module = 'src.experiments.tabular_nested_hpo'
    source = str(tmp_path / 'runner.py')
    original = 'd3c9313015420a74d3114057b0b90a60ec69ac41097b1ef2f65de7d715e0dc13'
    plan = {'sha256': {source: original}, 'seed': 42}
    freeze(tmp_path / 'plan.json', plan)
    digest = sha256_file(tmp_path / 'plan.json')
    monkeypatch.setattr(module + '.__file__', source)
    checks = []
    monkeypatch.setattr(module + '.require_hash', lambda path, sha: checks.append((path, sha)))
    revision = plan | {'sha256': {source: 'updated_source'}}
    freeze_plan(tmp_path, revision)
    freeze_plan(tmp_path, revision)
    assert checks == [(tmp_path / 'source/src/experiments/tabular_nested_hpo.py', original)] * 2
    assert sha256_file(tmp_path / 'plan.json') == digest
    with pytest.raises(ValueError, match='Frozen plan'):
        freeze_plan(tmp_path, revision | {'seed': 1})
    with pytest.raises(ValueError, match='Frozen plan'):
        freeze_plan(tmp_path, revision | {'sha256': {source: 'unexpected_source'}})
