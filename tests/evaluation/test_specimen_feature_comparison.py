"""A test-set perturbation must never change the primary feature-set assessment."""
import pandas as pd

from src.analysis.specimen_feature_comparison import (
    METRICS,
    validation_assessment,
    write_fusion_comparison,
)


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


def test_fusion_comparison_pairs_matching_models_and_ranks_false_alarms_after_critical_errors(tmp_path):
    fused, reference, comparison = (tmp_path / name for name in ('fused', 'reference', 'comparison'))
    fused.mkdir()
    reference.mkdir()
    metrics = dict.fromkeys(METRICS, 0.0)
    fixed = dict(metrics, model='hgb', critical_errors=5, false_alarms=4, critical_denominator=10)
    pd.DataFrame([fixed, dict(fixed, model='mlp', false_alarms=1)]).to_csv(
        fused / 'pooled_oof_metrics.csv', index=False,
    )
    pd.DataFrame([dict(fixed, split='test')]).to_csv(fused / 'holdout_metrics.csv', index=False)
    old_cv = {key: value for key, value in fixed.items() if key != 'critical_denominator'}
    pd.DataFrame([dict(old_cv, feature_set='manual52', critical_errors=6)]).to_csv(
        reference / 'cv_comparison.csv', index=False,
    )
    pd.DataFrame([dict(fixed, feature_set='tsfresh', split='test')]).to_csv(
        reference / 'holdout_comparison.csv', index=False,
    )

    write_fusion_comparison(fused, reference, comparison)

    ranked = pd.read_csv(comparison / 'ranked_results.csv')
    assert ranked.loc[ranked.split.eq('cv'), 'model'].tolist() == ['mlp', 'hgb', 'hgb']
    pairs = pd.read_csv(comparison / 'paired_deltas.csv')
    assert pairs.model.tolist() == ['hgb', 'hgb']
    assert pairs.delta_fusion_minus_reference_critical_errors.tolist() == [-1, 0]
