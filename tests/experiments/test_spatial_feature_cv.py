import numpy as np
import pandas as pd

from src.experiments.spatial_feature_cv import augment_features, select_arm


def test_spatial_preprocessing_ignores_evaluation_values_and_empty_fit_columns():
    spatial = pd.DataFrame({'variation': [-3., 1., np.nan, 10.],
                            'constant_on_fit': [4., 4., 4., 10.],
                            'empty_on_fit': [np.nan, np.nan, np.nan, 10.]})
    core_fit = np.array([[1., 1.], [0., 0.], [-1., -1.]])
    core_eval = np.array([[3., 1.]])

    fit, evaluated, stats = augment_features(core_fit, core_eval, spatial, [0, 1, 2], [3])
    other_fit, other_eval, other_stats = augment_features(
        core_fit, core_eval, spatial.assign(variation=[-3., 1., np.nan, -100.]), [0, 1, 2], [3],
    )

    assert stats == other_stats
    assert stats['columns'] == ['variation']
    assert stats['median'] == [-1.]
    np.testing.assert_array_equal(fit, other_fit)
    np.testing.assert_allclose(evaluated[:, :2], core_eval * np.sqrt(2 / 3))
    assert not np.array_equal(evaluated[:, 2], other_eval[:, 2])


def test_selection_balances_critical_errors_and_four_class_quality():
    truth = np.tile(np.repeat([0, 1, 2, 3], 50), 3)
    rows = pd.DataFrame({
        'label': truth, 'fold': np.repeat([0, 1, 2], 200),
        'specimen_id': np.repeat(['a', 'b', 'c'], 200), 'prediction': truth.copy(),
    })
    point = rows.copy()
    point.loc[50:59, 'prediction'] = 3
    tight = rows.copy()
    tight.loc[50:54, 'prediction'] = 3
    wide = rows.copy()
    wide['prediction'] = 2

    choice, trials = select_arm({'point_only': point, 'tight': tight, 'wide': wide})
    assert choice['arm'] == 'tight'
    assert set(trials.arm) == {'point_only', 'tight', 'wide'}
    fallback, _ = select_arm({'point_only': point, 'wide': wide})
    assert fallback['arm'] == 'point_only'
