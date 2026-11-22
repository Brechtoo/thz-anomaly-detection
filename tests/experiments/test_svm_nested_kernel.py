from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.svm import SVC

from src.experiments import svm_nested_kernel as kernel


@pytest.fixture
def fold_arrays():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(80, 9))
    evaluation = rng.normal(size=(22, 9))
    y = np.repeat(np.arange(4), [5, 17, 25, 33])
    return train, y, evaluation


@pytest.mark.parametrize(('disk', 'jobs'), [(False, 1), (True, 2)])
def test_reused_kernel_matches_direct_svc(tmp_path, monkeypatch, fold_arrays, disk, jobs):
    train, y, evaluation = fold_arrays
    if disk:
        monkeypatch.setattr(kernel, 'IN_MEMORY_BYTES', 0)
    candidates = [
        {'id': 'balanced', 'C': .3, 'gamma': .001, 'class_weight': 'balanced'},
        {'id': 'scale', 'C': .8, 'gamma': 'scale', 'class_weight': None},
        {'id': 'weighted', 'C': 4., 'gamma': .001, 'class_weight': {0: 1., 1: 3., 2: 1.5, 3: .75}},
    ]
    calls = []
    write_kernel = kernel._write_kernel

    def count_writes(target, left, fit, gamma):
        calls.append((len(left), gamma, isinstance(target, np.memmap)))
        return write_kernel(target, left, fit, gamma)

    monkeypatch.setattr(kernel, '_write_kernel', count_writes)
    records = list(kernel.iter_rbf_fits(train, y, evaluation, candidates, tmp_path, jobs=jobs))
    assert [result[0]['id'] for result in records] == ['balanced', 'weighted', 'scale']
    assert calls == [
        (len(left), gamma, disk)
        for gamma in (.001, 1 / (train.shape[1] * train.var()))
        for left in (train, evaluation)
    ]
    assert not list(tmp_path.iterdir())

    for candidate, predictions, scores, timings in records:
        direct = SVC(
            C=candidate['C'], gamma=candidate['gamma'], class_weight=candidate['class_weight'],
            decision_function_shape='ovr', random_state=42,
        ).fit(train, y)
        np.testing.assert_array_equal(direct.predict(evaluation), predictions)
        np.testing.assert_allclose(direct.decision_function(evaluation), scores, rtol=0, atol=1e-12)
        assert timings['support_vectors'] == direct.n_support_.sum()
        assert all(timings[key] >= 0 for key in ('training_seconds', 'prediction_seconds',
                                                 'train_kernel_seconds', 'evaluation_kernel_seconds'))


def test_partial_iteration_releases_scratch(tmp_path, monkeypatch, fold_arrays):
    monkeypatch.setattr(kernel, 'IN_MEMORY_BYTES', 0)
    candidates = [
        {'C': 1., 'gamma': .01, 'class_weight': None},
        {'C': 3., 'gamma': .01, 'class_weight': 'balanced'},
    ]
    fits = kernel.iter_rbf_fits(*fold_arrays, candidates, tmp_path)
    assert next(fits)[0] == candidates[0]
    assert list(tmp_path.glob('svm-nested-kernel-*/train.dat'))
    fits.close()
    assert not list(tmp_path.iterdir())


def test_reports_one_bad_candidate_and_continues(tmp_path, fold_arrays):
    candidates = [
        {'C': -1., 'gamma': .01, 'class_weight': None},
        {'C': 1., 'gamma': .01, 'class_weight': 'balanced'},
    ]
    failures = []
    fits = list(kernel.iter_rbf_fits(
        *fold_arrays, candidates, tmp_path, jobs=2,
        on_error=lambda candidate, error: failures.append((candidate, error)),
    ))
    assert [result[0] for result in fits] == [candidates[1]]
    assert len(failures) == 1
    assert failures[0][0] == candidates[0] and isinstance(failures[0][1], ValueError)


def test_refuses_incomplete_labels_and_checks_disk_budget(tmp_path, monkeypatch, fold_arrays):
    train, y, evaluation = fold_arrays
    candidate = {'C': 1., 'gamma': .01, 'class_weight': None}
    with pytest.raises(ValueError, match='all four training labels'):
        list(kernel.iter_rbf_fits(train, np.minimum(y, 2), evaluation, [candidate], tmp_path))

    monkeypatch.setattr(kernel, 'IN_MEMORY_BYTES', 0)
    monkeypatch.setattr(kernel.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    with pytest.raises(OSError, match='RBF kernel files need'):
        list(kernel.iter_rbf_fits(train, y, evaluation, [candidate], tmp_path))
    assert not list(tmp_path.iterdir())
