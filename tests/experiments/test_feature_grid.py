import numpy as np
import pandas as pd
import pytest

from src.data.split_manifest import sha256_file
from src.experiments import feature_grid
from src.experiments.feature_grid import (
    feature_combinations,
    fit_arrays,
    probabilities_in_label_order,
    rank,
)
from src.features.specimen_inputs import read_json, write_json


def test_grid_covers_all_subsets_except_combined_physical_versions():
    combos = feature_combinations()
    assert len(combos) == 23
    assert {len(sets) for sets in combos.values()} == {1, 2, 3, 4}
    assert all(not {'physical_v1', 'physical_v2'} <= set(sets) for sets in combos.values())


def test_probability_order_and_safety_rank():
    np.testing.assert_array_equal(
        probabilities_in_label_order([[.4, .1, .2, .3]], [3, 0, 1, 2]), [[.1, .2, .3, .4]],
    )
    with pytest.raises(ValueError, match='Invalid probability'):
        probabilities_in_label_order([[.4, .1, .2, .3]], [3, 0, 0, 2])
    items = {
        ('a', 'hgb'): pd.DataFrame({'label': [1, 3], 'prediction': [1, 2]}),
        ('b', 'hgb'): pd.DataFrame({'label': [1, 3], 'prediction': [3, 3]}),
    }
    assert rank(items)[0] == ('a', 'hgb')


def test_precomputed_rbf_exports_probabilities_without_using_evaluation_labels(tmp_path):
    rng = np.random.default_rng(42)
    y = np.repeat(np.arange(4), 10)
    train = y[:, None] + rng.normal(0, .04, (40, 3))
    prediction, probabilities, clocks, _ = fit_arrays(
        'rbf_svm', (train, y), train[:5], tmp_path,
    )
    assert prediction.shape == (5,) and probabilities.shape == (5, 4)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-8)
    assert clocks['training_seconds'] >= clocks['train_kernel_seconds']
    assert not list(tmp_path.iterdir())


def test_holdout_locks_one_selection_before_any_test_prediction(tmp_path, monkeypatch):
    membership = pd.DataFrame({
        'instance_id': [f'i{i}' for i in range(12)],
        'split': np.repeat(['train', 'validation', 'test'], 4),
        'label': list(range(4)) * 3,
    })
    combinations = {'manual': ['manual'], 'tsfel': ['tsfel']}
    write_json(tmp_path / 'shortlist.json', {'pairs': [[name, 'mlp'] for name in combinations]})

    class Grid:
        def holdout(self, names):
            values = np.full((4, 1), 0 if names == ['manual'] else 1)
            return values, values, values + 2, {}

    tests = []

    def fit(model, training, evaluation, scratch):
        marker = evaluation[0, 0]
        if marker >= 2:
            tests.append(read_json(scratch / 'test_selection.json')['pair'])
        prediction = np.arange(4)
        if marker == 0:
            prediction[1] = 3
        return prediction, np.eye(4), {}, []

    def checkpoint(path, model, combo, stage, parent, meta, features, fit):
        path.mkdir(parents=True)
        file = path / 'predictions.parquet'
        meta.assign(prediction=fit[0]).to_parquet(file, index=False)
        write_json(path / 'metrics.json', {'predictions_sha256': sha256_file(file)})

    monkeypatch.setattr(feature_grid, 'fit_arrays', fit)
    monkeypatch.setattr(feature_grid, 'checkpoint', checkpoint)
    feature_grid.run_holdouts(tmp_path, Grid(), membership, combinations, 'synthetic-run')
    assert tests == [['tsfel', 'mlp']]
    assert not (tmp_path / 'holdout/manual/mlp/test').exists()
