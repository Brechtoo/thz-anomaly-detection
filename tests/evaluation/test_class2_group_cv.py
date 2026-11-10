"""Class-set comparison must preserve groups, label semantics and preprocessing isolation."""
import numpy as np
import pandas as pd
import pytest

from src.experiments.anchor_group_cv import fold_pipeline
from src.experiments.class2_group_cv import (
    KNOWN,
    encode_labels,
    known_metrics,
    original_probabilities,
    restricted_prediction,
    training_indices,
)
from src.experiments.model_registry import create_model


def test_fixed_target_metrics_penalize_extra_output():
    y = [0, 1, 1, 3, 3]
    pred = [0, 2, 3, 1, 2]
    m = known_metrics(y, pred)
    assert m['recall_1'] == 0
    assert m['critical_1_count'] == 1 and m['critical_1_rate'] == .5
    assert m['false_alarm_count'] == 2 and m['false_alarm_rate'] == 1
    assert m['false_alarm_1_count'] == m['false_alarm_2_count'] == 1
    assert m['macro_f1'] == pytest.approx(1 / 3)
    assert m['balanced_accuracy'] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        known_metrics([2], [1])


def test_three_class_filter_precedes_preprocessing_and_preserves_group():
    data = pd.DataFrame(dict(fold=[0, 0, 0, 0, 1, 1, 1, 1],
                             label=[0, 1, 2, 3] * 2,
                             specimen_id=['a'] * 4 + ['b'] * 4))
    tr = training_indices(data, 1, KNOWN)
    np.testing.assert_array_equal(tr, [0, 1, 3])
    X = np.array([[0., np.nan], [2., 4.], [1e9, 1e9], [4., 8.],
                  [2e9, 2e9], [3e9, 3e9], [4e9, 4e9], [5e9, 5e9]])
    pipe = fold_pipeline(create_model('logistic_regression'))
    params = pipe.named_steps['model'].get_params()
    pipe.fit(X[tr], encode_labels(data.label.iloc[tr], KNOWN))
    np.testing.assert_array_equal(pipe.named_steps['imputer'].statistics_, [2, 6])
    np.testing.assert_array_equal(pipe.named_steps['scale'].mean_, [2, 6])
    assert params == pipe.named_steps['model'].get_params()
    assert set(data.specimen_id.iloc[tr]).isdisjoint(set(data.specimen_id[data.fold == 1]))
    p = original_probabilities(pipe, X, KNOWN)
    assert (p[:, 2] == 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    np.testing.assert_array_equal(KNOWN[pipe.predict(X)], np.argmax(p, axis=1))
    with pytest.raises(ValueError):
        encode_labels([2], KNOWN)


def test_output_restriction_is_not_label_merging():
    p = np.array([[.1, .2, .6, .1], [.1, .1, .5, .3]])
    np.testing.assert_array_equal(restricted_prediction(p), [1, 3])


def test_annotation_context_uses_only_same_body_known_cavity_labels():
    from src.analysis.class2_training_review import neighbor_context, positions
    data = pd.DataFrame(dict(instance_id=['s_x0_y0', 's_x1_y0', 's_x2_y0', 's_x3_y0', 's_x0_y1'],
                             specimen_id=['a', 'a', 'a', 'a', 'b'], label=[1, 2, 3, 0, 1]))
    data = pd.concat([data, positions(data)], axis=1)
    result = neighbor_context(data).iloc[0]
    assert result.instance_id == 's_x1_y0'
    assert result.neighbor_cavity_fraction == .5
    assert result.distance_to_1 == result.distance_to_3 == 1


def test_body_bootstrap_uses_same_target_metric_definitions(tmp_path):
    from src.analysis.class2_training_review import body_bootstrap
    data = pd.DataFrame(dict(instance_id=list('abcde'), specimen_id=['body'] * 5, label=[0, 1, 1, 3, 3]))
    directory = tmp_path / 'predictions'
    directory.mkdir()
    four, three = [0, 2, 3, 1, 2], [0, 1, 3, 1, 3]
    for arm, prediction in [('four_class', four), ('three_class', three)]:
        data.assign(prediction=prediction).to_csv(directory / f'{arm}__dummy.csv', index=False)
    body_bootstrap(tmp_path, data, ['dummy'], draws=20)
    result = pd.read_csv(tmp_path / 'paired_body_bootstrap.csv').set_index('metric')
    a, b = known_metrics(data.label, four), known_metrics(data.label, three)
    for key in result.index:
        assert result.loc[key, 'delta_median'] == pytest.approx(b[key] - a[key])
        assert result.loc[key, 'delta_p025'] == pytest.approx(result.loc[key, 'delta_p975'])
