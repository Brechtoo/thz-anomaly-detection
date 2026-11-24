import pandas as pd
import pytest

from src.analysis.spatial_feature_review import (
    CASE_BODIES,
    _new_axis_decisions,
    _plot_case_errors,
    _read_run,
    review_predictions,
)
from src.features.specimen_inputs import write_json


def test_review_counts_paired_changes_for_matched_hpo_and_prior_rescue(tmp_path):
    points = pd.DataFrame([
        ('a0', 'a', 0, 1, 0., 0.),
        ('a1', 'a', 0, 3, 1., 0.),
        ('b0', 'b', 1, 2, 0., 0.),
        ('b1', 'b', 1, 1, 1., 0.),
    ], columns=['instance_id', 'specimen_id', 'fold', 'label', 'x', 'y'])
    points = points.assign(feature_set='manual', source_dataset='source')
    decisions = {
        'point_only': [3, 3, 3, 1],
        'selected': [1, 3, 3, 1],
        'tight': [1, 3, 3, 1],
        'axis_tight': [1, 3, 2, 1],
        'prior_rescue': [3, 2, 2, 1],
        'hpo_same_radius_frozen': [3, 3, 3, 1],
        'hpo_spatial': [1, 3, 3, 3],
        'hpo_selected': [1, 3, 3, 3],
        'hpo_point_matched': [1, 3, 2, 1],
    }
    predictions = pd.concat([points.assign(arm=arm, prediction=values)
                             for arm, values in decisions.items()], ignore_index=True)
    tables = review_predictions(predictions.sample(frac=1, random_state=42), draws=20)

    matrix = tables['confusion'].query("arm == 'hpo_spatial' and scope == 'pooled'").set_index('label')
    assert matrix.loc[1, ['predicted_1', 'predicted_3']].tolist() == [1, 1]
    assert matrix.loc[0, [f'predicted_{label}' for label in range(4)]].sum() == 0
    changes = tables['paired_error_changes'].query("arm == 'hpo_spatial'")
    matched = changes.query("reference_arm == 'hpo_same_radius_frozen' and scope == 'pooled'")
    results = matched.set_index('error')[['support', 'retained', 'prevented', 'introduced']]
    assert results.loc['critical_1_or_2_to_3'].tolist() == [3, 1, 1, 1]
    assert results.loc['cavity_1_to_3'].tolist() == [2, 0, 1, 1]
    assert results.loc['uncertain_2_to_3'].tolist() == [1, 1, 0, 0]
    prior = changes.query("reference_arm == 'prior_rescue' and scope == 'specimen' and specimen_id == 'a'")
    assert prior.set_index('error').loc['false_alarm_3_to_1_or_2', 'prevented'] == 1
    near = changes.query("reference_arm == 'prior_rescue' and scope == 'diagnostic' "
                         "and slice == 'risk_adjacent_to_clear' and error == 'cavity_1_to_3'")
    assert near[['support', 'prevented']].values.tolist() == [[1, 1]]
    local_near = changes.query("reference_arm == 'prior_rescue' and scope == 'specimen_diagnostic' "
                               "and slice == 'risk_adjacent_to_clear' and error == 'cavity_1_to_3'")
    assert local_near[['specimen_id', 'support', 'prevented']].values.tolist() == [['a', 1, 1]]
    intervals = tables['paired_bootstrap'].query("metric == 'critical_error_rate'")
    delta = intervals.set_index(['arm', 'reference_arm']).delta
    assert delta.loc['hpo_spatial', 'hpo_same_radius_frozen'] == 0
    assert delta.loc['hpo_spatial', 'hpo_point_matched'] == pytest.approx(2 / 3)
    assert delta.loc['hpo_selected', 'selected'] == pytest.approx(1 / 3)
    assert delta.loc['axis_tight', 'tight'] == pytest.approx(-1 / 3)

    fixed = predictions.query("arm in ['point_only', 'tight', 'prior_rescue']")
    axis = predictions.query("arm in ['point_only', 'tight', 'prior_rescue', 'axis_tight']").copy()
    pd.testing.assert_frame_equal(_new_axis_decisions(axis, fixed), predictions.query("arm == 'axis_tight'"))
    axis.loc[axis.arm.eq('tight'), 'prediction'] = 0
    with pytest.raises(AssertionError, match='archived comparators'):
        _new_axis_decisions(axis, fixed)

    path = tmp_path / 'cases.png'
    cases = predictions.assign(feature_set='manual+tsfel',
                               specimen_id=predictions.specimen_id.map(dict(zip(('a', 'b'), CASE_BODIES))))
    _plot_case_errors(path, cases)
    assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_review_checks_local_predictions_and_plan_against_the_completed_run(tmp_path):
    run_id = '42'
    archived = tmp_path / 'mlartifacts' / run_id / 'artifacts'
    archived.mkdir(parents=True)
    write_json(tmp_path / 'run.json', {'run_id': run_id})
    write_json(tmp_path / 'plan.json', {'protocol': 'synthetic'})
    predictions = pd.DataFrame({'feature_set': ['manual'], 'arm': ['point_only']})
    predictions.to_parquet(tmp_path / 'predictions.parquet')
    for name in ('plan.json', 'predictions.parquet'):
        (archived / name).write_bytes((tmp_path / name).read_bytes())

    checked, source = _read_run(tmp_path, {'point_only'})
    pd.testing.assert_frame_equal(checked, predictions)
    assert source['run_id'] == run_id

    write_json(tmp_path / 'plan.json', {'protocol': 'modified'})
    with pytest.raises(ValueError, match='does not verify .*/plan.json'):
        _read_run(tmp_path, {'point_only'})
