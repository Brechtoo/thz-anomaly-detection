import numpy as np
import pandas as pd
import pytest

from src.experiments import svm_group_cv as experiment


def test_selection_prioritizes_critical_errors_over_macro_f1():
    truth = np.array([0, 1, 1, 2, 2, 3, 3])
    better_macro_f1 = [0, 1, 3, 2, 2, 3, 3]
    fewer_critical = [0, 1, 2, 2, 1, 2, 3]
    ranking = experiment.select_candidate(truth, [better_macro_f1, fewer_critical])

    assert ranking.candidate.tolist() == [1, 0]
    assert ranking.critical_errors.tolist() == [0, 1]
    assert ranking.iloc[0].macro_f1 < ranking.iloc[1].macro_f1


def test_tsfresh_recomputes_selection_with_only_fold_training_indices(tmp_path, monkeypatch):
    index = pd.Index([1, 3, 5, 7, 10, 12, 14, 16])
    labels = [0, 1, 2, 3] * 2
    train = pd.DataFrame({
        'instance_id': [f'measurement{i}' for i in index],
        'specimen_id': ['body0'] * 4 + ['body1'] * 4,
        'source_dataset': ['source'] * 8,
        'label': labels,
        'fold': [0] * 4 + [1] * 4,
    }, index=index)
    features = pd.DataFrame({'y__signal': labels}, index=index)
    calls = []

    def select(frame, y, train_index):
        calls.append(train_index.tolist())
        assert y.loc[train_index].tolist() == list(range(4))
        return frame

    monkeypatch.setattr(experiment, 'select_training_features', select)
    monkeypatch.setattr(experiment, 'GRID', [{'model__C': 10, 'model__gamma': 'scale'}])
    candidate, _ = experiment.tune(features, train, 'tsfresh', tmp_path, jobs=1)

    assert calls == [index[4:].tolist(), index[:4].tolist()]
    assert candidate == 0
    assert pd.read_csv(tmp_path / 'oof/predictions.csv').prediction.tolist() == labels


def test_manual_pipeline_imputes_and_scales_from_fit_rows_only():
    X_train = np.array([[np.nan, 8], [2, 8], [3, 8], [4, 8]])
    model = experiment.svm('manual52_early', {})
    model.fit(X_train, [0, 1, 2, 3])
    prediction = model.predict([[1000, 9], [np.nan, 9]])

    assert np.isin(prediction, [0, 1, 2, 3]).all()
    assert model['imputer'].statistics_ == pytest.approx([3, 8])
    assert model['constants'].get_support().tolist() == [True, False]
    assert model['scale'].mean_ == pytest.approx([3])


def test_shared_kernel_matches_direct_rbf_predictions(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(40, 9))
    X_val = rng.normal(size=(20, 9))
    X_train[:, -1] = X_val[:, -1] = 9
    y = np.tile(np.arange(4), 10)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(experiment.shared_rbf_candidates(X_train, X_val, y, pool, tmp_path,
                                                       grid=experiment.GRID, seed=experiment.SEED))

    assert sorted(i for i, _, _ in results) == list(range(len(experiment.GRID)))
    assert not list(tmp_path.iterdir())
    for i, prediction, _ in results:
        direct = experiment.svm('tsfresh', experiment.GRID[i]).fit(X_train, y)
        np.testing.assert_array_equal(prediction, direct.predict(X_val))
