import numpy as np
import pandas as pd
import pytest

from src.evaluation.spatial_postprocessing import candidate_grid
from src.experiments.spatial_postprocessing import (
    FAMILIES,
    MAX_LOSS,
    admissible,
    predict_outer,
    run,
    select_rules,
)


def test_guardrails_require_safety_gain_and_preserve_class_three_release():
    baseline = {
        'cavity_misses': 4, 'critical_errors': 9, 'macro_f1': .8, 'balanced_accuracy': .8,
        'recall_1': .8, 'recall_3': .9, 'precision_3': .9, 'release_rate': .7,
        'false_alarm_rate': .1, 'poor_measurement_release_rate': .1,
    }
    clear = np.array([.85, .9, .95])
    candidate = {**baseline, 'cavity_misses': 3, 'critical_errors': 8}
    assert admissible(candidate, baseline, clear, clear)
    assert not admissible(baseline, baseline, clear, clear)
    assert not admissible({**candidate, 'critical_errors': 10}, baseline, clear, clear)
    assert not admissible({**candidate, 'release_rate': .7 - MAX_LOSS['release_rate'] - .01},
                          baseline, clear, clear)
    assert not admissible(candidate, baseline, np.array([.60, .9, .95]), clear)


def points(specimen, fold):
    labels = np.repeat([0, 1, 2, 3], [1, 2, 12, 8])
    probabilities = np.eye(4)[labels] * .96 + .01
    probabilities[2] = [.03, .40, .09, .48]
    return pd.DataFrame({
        'instance_id': [f'{specimen}-{i}' for i in range(len(labels))],
        'specimen_id': specimen, 'source_dataset': 'source', 'fold': fold,
        'label': labels, 'prediction': probabilities.argmax(axis=1),
        'x': np.arange(len(labels), dtype=float), 'y': 0.,
        **{f'p_{label}': probabilities[:, label] for label in range(4)},
    })


def test_frozen_selection_applies_to_same_outer_points_without_reading_outer_labels():
    candidates = tuple(candidate_grid())
    inner = pd.concat([points(f'inner-{fold}', fold) for fold in range(3)], ignore_index=True)
    selected, trials = select_rules(inner, candidates)
    assert selected['pointwise'] is not None
    assert len(trials) == len(candidates) + 1

    outer = points('outer', 0)
    result, rules = predict_outer(outer, 'manual', selected, candidates, trials)
    changed, other_rules = predict_outer(outer.assign(label=3), 'manual', selected, candidates, trials)
    pd.testing.assert_frame_equal(result.drop(columns='label'), changed.drop(columns='label'))
    assert rules == other_rules
    assert set(result.strategy) == {'baseline', 'selected_spatial', *FAMILIES}
    for _, arm in result.groupby('strategy'):
        np.testing.assert_array_equal(arm.instance_id, outer.instance_id)
        np.testing.assert_array_equal(arm.baseline_prediction, outer.prediction)
    assert result.query("strategy == 'pointwise'").iloc[2].prediction == 2


def test_run_refuses_to_overwrite_a_historical_or_existing_directory(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    with pytest.raises(ValueError, match='outside'):
        run(source, source / 'spatial')
    known = tmp_path / 'existing'
    known.mkdir()
    marker = known / 'existing.txt'
    marker.write_text('preserve')
    with pytest.raises(FileExistsError):
        run(source, known)
    assert marker.read_text() == 'preserve'
