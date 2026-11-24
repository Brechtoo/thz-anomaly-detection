import pandas as pd
import pytest

from src.analysis.spatial_feature_report import summarize_spatial_features


def predictions():
    points = pd.DataFrame([
        ('a0', 'a', 0, 1, 0., 0.),
        ('a1', 'a', 0, 3, 1., 0.),
        ('a2', 'a', 0, 0, 2., 0.),
        ('b0', 'b', 1, 2, 0., 0.),
        ('b1', 'b', 1, 3, 1., 0.),
        ('b2', 'b', 1, 1, 2., 0.),
    ], columns=['instance_id', 'specimen_id', 'fold', 'label', 'x', 'y'])
    points = points.assign(feature_set='manual', source_dataset='source')
    decisions = {
        'point_only': [3, 3, 0, 3, 3, 1],
        'prior_native': [3, 3, 0, 3, 1, 3],
        'prior_rescue': [3, 2, 0, 3, 1, 1],
        'tight': [1, 3, 0, 2, 3, 3],
    }
    return pd.concat([points.assign(arm=arm, prediction=prediction)
                      for arm, prediction in decisions.items()], ignore_index=True)


def test_paired_reports_use_the_same_points_and_expected_comparators():
    frame = predictions().sample(frac=1, random_state=42)
    tables = summarize_spatial_features(frame, draws=30)
    pooled = tables['pooled'].set_index('arm')
    assert pooled.loc['point_only', 'critical_errors'] == 2
    assert pooled.loc['tight', 'critical_errors'] == 1
    assert pooled.loc['tight', 'prevented_critical_errors'] == 2
    assert pooled.loc['tight', 'introduced_critical_errors'] == 1
    assert pooled.loc['tight', 'delta_critical_error_rate'] == pytest.approx(-1 / 3)
    assert set(tables['specimen'].specimen_id) == {'a', 'b'}
    assert set(tables['fold'].fold) == {0, 1}
    assert 'risk_adjacent_to_clear' in set(tables['diagnostic']['slice'])
    intervals = tables['paired_bootstrap'].query("arm == 'tight' and metric == 'critical_error_rate'")
    delta = intervals.set_index('reference_arm').delta.to_dict()
    assert delta == pytest.approx({'point_only': -1 / 3, 'prior_native': -2 / 3, 'prior_rescue': -1 / 3})
    repaired = tables['prior_rescue_errors'].query("arm == 'tight'").set_index('historical_error')
    assert repaired.loc['cavity_release_1_to_3', 'correct'] == 1
    assert repaired.loc['uncertain_release_2_to_3', 'predicted_2'] == 1
    assert repaired.loc['false_alarm_3_to_1_or_2', 'correct'] == 2


@pytest.mark.parametrize('change', ['duplicate', 'missing_control', 'mismatched_coordinates'])
def test_mismatched_pairs_fail_closed(change):
    frame = predictions()
    if change == 'duplicate':
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif change == 'missing_control':
        frame = frame.iloc[1:]
    else:
        frame.loc[frame.arm.eq('tight') & frame.instance_id.eq('a0'), 'x'] = 7.
    with pytest.raises(ValueError, match='unique complete|matched point_only|measurements differ'):
        summarize_spatial_features(frame, draws=5)


def test_feature_sets_must_match_the_frozen_support_and_folds():
    base = predictions()
    shifted = base.assign(feature_set='combined', x=base.x + 1.)
    with pytest.raises(ValueError, match='same measured points'):
        summarize_spatial_features(pd.concat([base, shifted]), draws=5)
