import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.evaluation.metrics import calculate_error_counts
from src.evaluation.svm_nested_selection import (
    GUARDRAILS,
    SCORE_COLUMNS,
    apply_policy,
    body_cross_fitted_probabilities,
    calibrated_trials,
    choose_policy,
    decision_metrics,
    raw_metrics,
    review_metrics,
)


@pytest.fixture
def inner_oof():
    rng = np.random.default_rng(100)
    labels = np.tile(np.arange(4), 9)
    scores = np.ones((len(labels), 4)) * np.eye(4)[labels] + rng.normal(scale=.25, size=(len(labels), 4))
    return pd.DataFrame(scores, columns=SCORE_COLUMNS).assign(
        label=labels, prediction=scores.argmax(axis=1), fold=np.repeat(np.arange(3), 12),
        specimen_id=np.repeat(np.arange(9), 4),
    )


@pytest.mark.parametrize('method', ['multinomial', 'ovr'])
def test_cross_fitted_calibration_excludes_held_body_and_other_folds(inner_oof, method):
    held = inner_oof.specimen_id.eq(0)
    original = body_cross_fitted_probabilities(inner_oof, method)
    changed = inner_oof.copy()
    changed.loc[held, 'label'] = 0
    # These scores come from different SVM fits and can depend on the held specimen.
    changed.loc[changed.fold.ne(0), SCORE_COLUMNS] *= 10
    altered = body_cross_fitted_probabilities(changed, method)
    np.testing.assert_array_equal(original[held], altered[held])
    np.testing.assert_allclose(original.sum(axis=1), 1)
    with pytest.raises(ValueError, match='more than one'):
        body_cross_fitted_probabilities(inner_oof.assign(
            fold=inner_oof.fold.mask(inner_oof.index == 0, 1),
        ), method)


def test_policy_preserves_non_green_labels_and_distinguishes_review():
    hard = np.array([0, 3, 3, 3])
    probabilities = np.array([[.06, .02, .02, .90], [.10, .20, .15, .55],
                              [.30, .08, .02, .60], [.1, .1, .1, .7]])
    y = np.array([0, 1, 3, 2])
    policy = {'gate': 'predict', 'threshold': .6, 'fallback': 'argmax012'}
    predicted = apply_policy(hard, probabilities, policy)
    np.testing.assert_array_equal(predicted, [0, 1, 3, 3])
    np.testing.assert_array_equal(
        apply_policy(hard, probabilities, policy | {'gate': 'argmax'}), [3, 1, 3, 3],
    )
    review = apply_policy(hard, probabilities, policy | {'fallback': 'review'})
    result = review_metrics(y, review)
    assert result['critical_1_errors'] == 0
    assert result['critical_2_errors'] == 1
    assert result['review_count'] == 1
    assert result['false_alarms'] == 0
    with pytest.raises(ValueError, match=r'in \[0, 3\]'):
        decision_metrics(y, review)


def test_four_class_metrics_match_reference():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    predicted = np.array([0, 3, 0, 3, 3, 1, 2, 3])
    metrics = decision_metrics(y, predicted)
    for key, reference in calculate_error_counts(y, predicted).items():
        assert metrics[key] == reference
    assert metrics['macro_f1'] == pytest.approx(f1_score(y, predicted, average='macro'))
    assert metrics['balanced_accuracy'] == pytest.approx(balanced_accuracy_score(y, predicted))
    for label in np.arange(4):
        assert metrics[f'recall_{label}'] == recall_score(y, predicted, average=None)[label]
        assert metrics[f'precision_{label}'] == precision_score(y, predicted, average=None)[label]


def test_selection_enforces_four_classes_and_records_infeasibility(inner_oof):
    reference = raw_metrics(inner_oof)
    raw = reference | {'candidate_id': 'raw', 'critical_error_rate': 4., 'safety_score': 6.}
    valid = raw | {'candidate_id': 'valid', 'threshold': .85, 'critical_error_rate': 1.9, 'safety_score': 4.}
    trivial = valid | {'candidate_id': 'trivial', 'threshold': .99, 'critical_error_rate': 0.,
                       'safety_score': 0., 'recall_3': 0., 'worst_fold_recall_3': 0.}
    choice = choose_policy(pd.DataFrame([raw, trivial, valid]), reference)
    assert choice['candidate_id'] == 'valid'
    assert choice['feasible']
    assert choice['below_target']
    assert choice['floors']['macro_f1'] == pytest.approx(reference['macro_f1'] - .04)
    json.dumps(choice, allow_nan=False)
    infeasible = raw | {'recall_0': GUARDRAILS['recall_0'] - .01}
    fallback = choose_policy(pd.DataFrame([trivial, infeasible]))
    assert fallback['candidate_id'] == 'raw'
    assert not fallback['feasible']
    assert fallback['guardrails_missed'] == ['recall_0']


def test_calibrated_search_preserves_hard_and_scores_both_methods(inner_oof):
    baseline = raw_metrics(inner_oof)
    ranking = calibrated_trials(inner_oof)
    assert set(ranking.calibration) == {'multinomial', 'ovr'}
    assert ranking.calibration_log_loss.gt(0).all()
    original = ranking.query("gate == 'predict' and threshold == 0")
    assert original.critical_errors.eq(baseline['critical_errors']).all()
    assert original.macro_f1.eq(baseline['macro_f1']).all()
    assert set(ranking.fallback) == {'argmax012', 'class2'}
    assert set(ranking.gate) == {'predict', 'argmax'}
