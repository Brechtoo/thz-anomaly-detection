import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.analysis.spatial_postprocessing_report import spatial_metrics, summarize
from src.evaluation.metrics import calculate_error_counts


def test_metrics_match_project_error_definitions_and_release_semantics():
    truth = [0, 1, 1, 2, 2, 3, 3, 3]
    prediction = [3, 3, 1, 3, 0, 2, 0, 3]

    result = spatial_metrics(truth, prediction)
    project = calculate_error_counts(truth, prediction)

    for key in ('critical_errors', 'false_alarms', 'overall_unsafe_clearances'):
        assert result[key] == project[key]
    for key in ('critical_error_rate', 'false_alarm_rate', 'overall_unsafe_clearance_rate'):
        assert result[key] == pytest.approx(project[key] / 100)
    assert result['cavity_misses'] == result['uncertain_misses'] == 1
    assert result['cavity_miss_rate'] == result['uncertain_miss_rate'] == .5
    assert result['poor_measurement_releases'] == 1
    assert result['release_rate'] == .5
    assert result['critical_given_release_rate'] == .5
    assert result['recall_3'] == pytest.approx(1 / 3)
    assert result['precision_3'] == .25
    assert result['f1_3'] == pytest.approx(2 / 7)
    assert result['true3_nonrelease_rate'] == pytest.approx(2 / 3)
    assert result['macro_f1'] == pytest.approx(f1_score(truth, prediction, labels=[0, 1, 2, 3], average='macro'))
    assert result['balanced_accuracy'] == pytest.approx(balanced_accuracy_score(truth, prediction))


def test_missing_and_invalid_classes():
    result = spatial_metrics([3], [3])
    assert result['macro_f1'] == .25
    assert result['balanced_accuracy'] == 1
    assert np.isnan(result['critical_error_rate'])
    with pytest.raises(ValueError, match='four classes'):
        spatial_metrics([3], [-1])


def paired_points():
    # The three sources overlap in coordinate space. A is a singleton cavity
    # at the edge of a sampled region, B has a two-point cavity component,
    # and C is an isolated measurement with no geometry for neighbor evidence.
    baseline = pd.DataFrame(
        [
            ('a0', 's1', 'a', 0, 0., 0., 1, 3),
            ('a1', 's1', 'a', 0, 1., 0., 3, 3),
            ('a2', 's1', 'a', 0, 2., 0., 3, 3),
            ('b0', 's2', 'b', 1, 0., 0., 1, 1),
            ('b1', 's2', 'b', 1, 1., 0., 1, 1),
            ('b2', 's2', 'b', 1, 2., 0., 2, 3),
            ('c0', 's3', 'c', 2, 0., 0., 3, 3),
        ],
        columns=['instance_id', 'source_dataset', 'specimen_id', 'fold', 'x', 'y', 'label', 'prediction'],
    )
    baseline = baseline.assign(baseline='manual', strategy='baseline', baseline_prediction=baseline.prediction)
    spatial = baseline.assign(strategy='rescue', prediction=[1, 2, 3, 1, 1, 3, 3])
    return pd.concat([baseline, spatial], ignore_index=True)


def test_summarize_pairs_decisions_and_keeps_geometry_local():
    frame = paired_points().sample(frac=1, random_state=42)
    tables = summarize(frame, draws=20)

    baseline, rescue = tables['pooled'].itertuples(index=False)
    assert baseline.n == rescue.n == 7
    assert baseline.critical_errors == 2
    assert rescue.critical_errors == 1
    assert rescue.prevented_critical_errors == 1
    assert rescue.introduced_critical_errors == 0
    assert rescue.false_alarms == 1
    assert len(tables['fold']) == len(tables['specimen']) == 6

    diagnostic = tables['diagnostic'].query("strategy == 'rescue'").set_index('slice')
    assert diagnostic.loc['cavity_singleton_with_neighbors', 'n'] == 1
    assert diagnostic.loc['cavity_singleton_with_neighbors', 'prevented_critical_errors'] == 1
    assert diagnostic.loc['cavity_small_component_2_to_4', 'n'] == 2
    assert diagnostic.loc['risk_adjacent_to_clear', 'n'] == 1
    assert diagnostic.loc['clear_adjacent_to_risk', 'n'] == 1
    assert diagnostic.loc['no_neighbors', 'n'] == 1
    interval = tables['paired_bootstrap'].query("metric == 'critical_error_rate'").iloc[0]
    assert interval.delta == -.25
    assert interval.specimens == 3
    assert interval.lower_95 <= interval.upper_95 <= 0
    pd.testing.assert_frame_equal(tables['paired_bootstrap'], summarize(frame, draws=20)['paired_bootstrap'])


def test_summarize_rejects_incomplete_or_changed_pairing():
    frame = paired_points()
    with pytest.raises(ValueError, match='measurements differ'):
        summarize(frame.iloc[:-1], draws=5)
    frame.loc[0, 'baseline_prediction'] = 2
    with pytest.raises(ValueError, match='untouched reference'):
        summarize(frame, draws=5)


def test_pointwise_pairing_uses_the_pointwise_decisions():
    frame = paired_points()
    pointwise = frame[frame.strategy == 'baseline'].assign(
        strategy='pointwise', prediction=[2, 3, 3, 1, 1, 3, 3],
    )
    tables = summarize(pd.concat([frame, pointwise]), draws=20)
    row = tables['paired_pointwise'].query("strategy == 'rescue' and metric == 'critical_error_rate'").iloc[0]
    assert row.delta == row.lower_95 == row.upper_95 == 0.
