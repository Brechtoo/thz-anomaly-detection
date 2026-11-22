import numpy as np
import pandas as pd
import pytest

from src.analysis.rbf_svm_oof_thresholds import (
    PROBABILITIES,
    decision_metrics,
    decision_tables,
    reliability,
    route,
    threshold_below_target,
)


@pytest.mark.parametrize(('strategy', 'fallback', 'false_alarms', 'reviews'), [
    ('review', [-1, -1, -1], 1, 3),
    ('class2', [2, 2, 2], 2, 0),
    ('argmax012', [1, 2, 0], 1, 0),
])
def test_release_keeps_boundary_and_separates_review_from_false_alarms(strategy, fallback, false_alarms, reviews):
    hard = np.array([3, 3, 3, 3, 1])
    y = np.array([1, 2, 3, 3, 3])
    p = np.array([[.1, .5, .1, .3], [.1, .1, .5, .3], [.5, .1, .1, .3],
                  [.1, .1, .1, .7], [.1, .1, .1, .7]])
    actual = route(hard, p, .7, strategy)
    np.testing.assert_array_equal(actual, [*fallback, 3, 1])
    metrics = decision_metrics(y, actual, hard)
    assert metrics['critical_errors'] == 0
    assert metrics['false_alarms'] == false_alarms
    assert metrics['review'] == reviews
    assert metrics['true3_nonclear'] == 2
    assert metrics['newly_cleared_vs_predict'] == 0


def test_strict_target_respects_ties_and_uses_smallest_threshold():
    y = np.full(100, 1)
    hard = np.array([3, 3, 3, *([1] * 97)])
    scores = np.array([.90, .80, .80, *([.30] * 97)])
    threshold = threshold_below_target(y, hard, scores)
    assert threshold == np.nextafter(.8, np.inf)
    assert ((hard == 3) & (scores >= threshold)).sum() == 1
    assert ((hard == 3) & (scores >= np.nextafter(threshold, 0))).sum() == 3
    assert threshold_below_target(y, y, scores) == 0
    with pytest.raises(ValueError, match='true label'):
        threshold_below_target(hard + 10, hard, scores)


def test_group_target_and_argmax_differences():
    data = pd.DataFrame({**dict(zip(PROBABILITIES, [
        [.04, .05, .02, .07], [.03, .30, .46, .07], [.03, .05, .07, .06], [.90, .60, .45, .80],
    ], strict=True)), 'fold': [0, 0, 1, 1], 'specimen_id': ['a', 'a', 'b', 'b'],
        'label': [1, 3, 2, 3], 'prediction': [3, 2, 3, 3]})
    _, points, labels = decision_tables({'example': data})
    folds = points.query("point == 'all_folds_lt2' and scope == 'fold'")
    assert folds.critical_errors.eq(0).all()
    hard = points.query("gate == 'predict' and routing == 'review' and point == 'baseline' and scope == 'pooled'")
    alt = points.query("gate == 'argmax' and routing == 'review' and point == 'baseline' and scope == 'pooled'")
    assert hard.critical_errors.item() == 2
    assert alt.critical_errors.item() == 1
    assert alt.newly_cleared_vs_predict.item() == 1
    assert labels[['pred_review', *[f'pred_{i}' for i in range(4)]]].sum(axis=1).equals(labels.support)
    missing = decision_metrics(np.array([0]), np.array([0]), np.array([0]))
    assert np.isnan(missing['critical_error_rate']) and np.isnan(missing['false_alarm_rate'])


def test_reliability_last_bin_includes_probability_one():
    bins, ece = reliability(np.array([0.0, .08, .91, 1.0]), np.array([False, True, True, True]))
    assert [row['bin'] for row in bins] == [0, 9]
    assert [row['n'] for row in bins] == [2, 2]
    assert ece == pytest.approx(.2525)
