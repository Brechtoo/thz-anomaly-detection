"""A test-set perturbation must never change the primary feature-set assessment."""
import pandas as pd

from src.analysis.specimen_feature_comparison import METRICS, validation_assessment


def test_validation_conclusion_ignores_test_metrics_and_row_order():
    manual = pd.DataFrame([{'model': 'hgb', 'split': split, **{key: value for key in METRICS}}
                           for split, value in [('validation', .4), ('test', .1)]])
    tsfresh = manual.copy()
    tsfresh.loc[tsfresh.split.eq('validation'), METRICS] = .3
    before, conclusion = validation_assessment(manual, tsfresh)
    assert conclusion['preferred_for_critical_error_priority'] == 'tsfresh708'
    manual.loc[manual.split.eq('test'), METRICS] = 1e9
    tsfresh.loc[tsfresh.split.eq('test'), METRICS] = -1e9
    after, other = validation_assessment(manual.sample(frac=1, random_state=3), tsfresh)
    pd.testing.assert_frame_equal(before, after)
    assert conclusion == other and other['selection_uses_test'] is False
