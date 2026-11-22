"""Training-only calibration and four-class selection for the nested HeRAI RBF grid."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import calculate_error_counts, critical_breakdown

LABELS = np.arange(4)
SCORE_COLUMNS = [f'score_{label}' for label in LABELS]
CALIBRATIONS = ('multinomial', 'ovr')
THRESHOLDS = (.45, .5, .55, .6, .65, .7, .75, .8, .85, .875, .9, .925, .95, .97, .99)
FALLBACKS = ('argmax012', 'class2')
GUARDRAILS = {
    'macro_f1': .60, 'balanced_accuracy': .64,
    'recall_0': .45, 'recall_1': .55, 'recall_2': .15, 'recall_3': .55,
    'precision_3': .82, 'worst_fold_recall_3': .45,
}
BASELINE_TOLERANCE = {'macro_f1': .04, 'balanced_accuracy': .04, 'recall_3': .15}
CRITICAL_TARGET = 2.
BODY_SUPPORT_SCALE = 20.
SAFETY_WEIGHTS = {
    'critical_error_rate': 1., 'critical_1_rate': 1.,
    'critical_specimen_stabilized_rate': .25, 'critical_worst_fold_rate': .10,
}


def fit_calibrator(scores, y, method='multinomial'):
    """Fit on eligible grouped SVM evaluation scores in class order 0, 1, 2, 3."""
    if set(np.unique(y)) != set(LABELS):
        raise ValueError('Calibration training requires observations of all four HeRAI classes')
    if not np.isfinite(scores).all() or np.shape(scores) != (len(y), len(LABELS)):
        raise ValueError('Calibration requires finite four-class scores')
    logistic = LogisticRegression(C=1., solver='lbfgs', max_iter=500, random_state=42)
    classifier = {'multinomial': logistic, 'ovr': OneVsRestClassifier(logistic, n_jobs=1)}[method]
    return make_pipeline(StandardScaler(), classifier).fit(scores, y)


def predict_probabilities(calibrator, scores):
    classes = calibrator.classes_
    probabilities = calibrator.predict_proba(scores)[:, np.argsort(classes)]
    if not np.array_equal(np.sort(classes), LABELS) or not np.isfinite(probabilities).all():
        raise ValueError('Calibration returned invalid four-class probabilities')
    np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=1e-10, rtol=0.)
    return probabilities


def body_cross_fitted_probabilities(frame, method='multinomial'):
    """Calibrate each assessed body using peer bodies assessed by the same inner SVM.

    Across different assessment folds, OOF scores could be from SVMs trained on
    the assessed body. Using only scores from its own fold excludes that path.
    The final outer calibrator can consume all inner OOF rows because none of
    those row labels or their SVM fits involve the outer assessment bodies.
    """
    scores = frame[SCORE_COLUMNS].to_numpy()
    y, body = frame.label.to_numpy(), frame.specimen_id.to_numpy()
    if frame[['fold', 'specimen_id', 'label']].isna().any().any():
        raise ValueError('Calibration needs complete fold, body and label assignments')
    membership = frame[['specimen_id', 'fold']].drop_duplicates()
    if not membership.specimen_id.is_unique:
        raise ValueError('A calibration body occurs in more than one inner assessment fold')
    probabilities = np.empty_like(scores, dtype=float)
    for indices in frame.groupby('fold', sort=False).indices.values():
        groups = body[indices]
        for assessed in np.unique(groups):
            fit = indices[groups != assessed]
            predict = indices[groups == assessed]
            calibrator = fit_calibrator(scores[fit], y[fit], method)
            probabilities[predict] = predict_probabilities(calibrator, scores[predict])
    return probabilities


def apply_policy(hard, probabilities, policy):
    """Gate only outgoing decisions of class 3 and preserve other base labels."""
    gate, threshold = policy['gate'], policy['threshold']
    if gate == 'predict':
        prediction = np.asarray(hard, dtype=int).copy()
    elif gate == 'argmax':
        prediction = probabilities.argmax(axis=1)
    else:
        raise ValueError(f'Unknown gate {gate}')
    rejected = (prediction == 3) & (probabilities[:, 3] < threshold)
    if policy['fallback'] == 'argmax012':
        prediction[rejected] = probabilities[rejected, :3].argmax(axis=1)
    else:
        prediction[rejected] = {'class2': 2, 'review': -1}[policy['fallback']]
    return prediction


def decision_metrics(y, prediction):
    """Score primary predictions as four classes, with original HeRAI error definitions."""
    y, prediction = np.asarray(y, dtype=int), np.asarray(prediction, dtype=int)
    if y.ndim != 1 or y.shape != prediction.shape or not y.size:
        raise ValueError('Four-class predictions need matching nonempty vectors')
    if ((y < 0) | (y > 3) | (prediction < 0) | (prediction > 3)).any():
        raise ValueError('Four-class metrics require labels and predictions in [0, 3]')
    counts = calculate_error_counts(y, prediction) | critical_breakdown(y, prediction)
    matrix = np.bincount(y * 4 + prediction, minlength=16).reshape(4, 4)
    true, predicted, correct = matrix.sum(axis=1), matrix.sum(axis=0), np.diag(matrix)
    recall = np.divide(correct, true, out=np.zeros(4), where=true != 0)
    precision = np.divide(correct, predicted, out=np.zeros(4), where=predicted != 0)
    f1 = np.divide(2 * correct, true + predicted, out=np.zeros(4), where=true + predicted != 0)
    return counts | {
        'n': len(y), 'accuracy': float(correct.sum() / len(y)),
        'macro_f1': float(f1.mean()), 'balanced_accuracy': float(recall[true > 0].mean()),
        'critical_support': int(true[1] + true[2]), 'class3_support': int(true[3]),
        'release_rate': float(100 * predicted[3] / len(y)),
        **{f'recall_{label}': float(recall[label]) for label in LABELS},
        **{f'precision_{label}': float(precision[label]) for label in LABELS},
        **{f'f1_{label}': float(f1[label]) for label in LABELS},
        **{f'support_{label}': int(true[label]) for label in LABELS},
    }


def review_metrics(y, prediction):
    """Report extra review decisions separately from four-class classification."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    if not np.isin(prediction, [-1, *LABELS]).all():
        raise ValueError('Unknown review outcome')
    n, no_cavity = len(y), int((y == 3).sum())
    return calculate_error_counts(y, prediction) | critical_breakdown(y, prediction) | {
        'n': n, 'review_count': int((prediction == -1).sum()),
        'review_rate': float(100 * (prediction == -1).sum() / n),
        'release_rate': float(100 * (prediction == 3).sum() / n),
        'class3_review_count': int(((prediction == -1) & (y == 3)).sum()),
        'class3_review_rate': float(100 * ((prediction == -1) & (y == 3)).sum() / no_cavity) if no_cavity else np.nan,
        'class3_release_rate': float(100 * ((prediction == 3) & (y == 3)).sum() / no_cavity) if no_cavity else np.nan,
    }


def _group_counts(y, codes, prediction):
    bins = np.bincount(codes * 16 + y * 4 + prediction, minlength=(int(codes.max()) + 1) * 16)
    matrices = bins.reshape(-1, 4, 4)
    critical_support = matrices[:, 1:3, :].sum(axis=(1, 2))
    critical_count = matrices[:, 1:3, 3].sum(axis=1)
    true3 = matrices[:, 3, :].sum(axis=1)
    return critical_support, critical_count, true3, matrices[:, 3, 3]


def _grouped_metrics(y, prediction, fold_codes, body_codes):
    metrics = decision_metrics(y, prediction)
    body_support, body_critical, _, _ = _group_counts(y, body_codes, prediction)
    fold_support, fold_critical, true3, correct3 = _group_counts(y, fold_codes, prediction)
    body_eligible = body_support > 0
    body_weights = body_support[body_eligible] / (body_support[body_eligible] + BODY_SUPPORT_SCALE)
    body_rates = 100 * body_critical[body_eligible] / body_support[body_eligible]
    group_metrics = {
        'critical_specimen_mean_rate': float(body_rates.mean()),
        'critical_specimen_stabilized_rate': float(np.average(body_rates, weights=body_weights)),
        'critical_worst_fold_rate': float(np.max(100 * fold_critical[fold_support > 0] / fold_support[fold_support > 0])),
        'worst_fold_recall_3': float(np.min(correct3[true3 > 0] / true3[true3 > 0])),
    }
    metrics.update(group_metrics)
    metrics['safety_score'] = sum(metrics[key] * weight for key, weight in SAFETY_WEIGHTS.items())
    return metrics


def _inputs(frame):
    if frame.empty or frame[['label', 'prediction', 'fold', 'specimen_id']].isna().any().any():
        raise ValueError('Selection requires complete inner OOF labels, predictions, bodies and folds')
    y = frame.label.to_numpy(dtype=int)
    if not np.array_equal(np.unique(y), LABELS):
        raise ValueError('Selection requires all four HeRAI classes')
    return (y, frame.prediction.to_numpy(dtype=int), pd.factorize(frame.fold, sort=True)[0],
            pd.factorize(frame.specimen_id, sort=True)[0])


def raw_metrics(frame):
    """Measure the native hard SVM decision before calibrating a shortlisted model."""
    y, hard, fold_codes, body_codes = _inputs(frame)
    return {'gate': 'predict', 'threshold': 0., 'fallback': 'argmax012', 'calibration': 'multinomial',
            **_grouped_metrics(y, hard, fold_codes, body_codes)}


def calibrated_trials(frame):
    """Explore calibrators and restrained gates on the inner assessment predictions."""
    y, hard, fold_codes, body_codes = _inputs(frame)
    trials = []
    for method in CALIBRATIONS:
        probabilities = body_cross_fitted_probabilities(frame, method)
        calibration = {
            'calibration': method,
            'calibration_log_loss': float(log_loss(y, probabilities, labels=LABELS)),
            'calibration_brier': float(np.sum((probabilities - (y[:, None] == LABELS)) ** 2, axis=1).mean()),
        }
        for gate in ('predict', 'argmax'):
            for threshold in (0., *THRESHOLDS):
                for fallback in FALLBACKS if threshold else ('argmax012',):
                    policy = {'gate': gate, 'threshold': threshold, 'fallback': fallback}
                    prediction = apply_policy(hard, probabilities, policy)
                    trials.append(calibration | policy | _grouped_metrics(y, prediction, fold_codes, body_codes))
    return pd.DataFrame(trials)


def choose_policy(trials, baseline=None):
    """Apply quality guardrails, then prefer safety, four-class quality and false alarms."""
    if trials.empty:
        raise ValueError('No successful inner OOF trial is available')
    floors = GUARDRAILS.copy()
    if baseline is not None:
        floors.update({key: max(floors[key], baseline[key] - tolerance)
                       for key, tolerance in BASELINE_TOLERANCE.items()})
    margins = trials[list(floors)].sub(pd.Series(floors))
    ranking = trials.assign(
        feasible=margins.ge(-1e-12).all(axis=1),
        guardrail_deficit=(-margins).clip(lower=0.).sum(axis=1),
        below_target=trials.critical_error_rate.lt(CRITICAL_TARGET),
        classification_quality=(trials.macro_f1 + trials.balanced_accuracy) / 2,
    )
    best = ranking.loc[ranking.feasible]
    priorities = ['below_target', 'safety_score', 'classification_quality', 'false_alarm_rate']
    ascending = [False, True, False, True]
    if best.empty:
        regular = ranking.loc[ranking.gate.eq('predict') & ranking.threshold.eq(0)]
        best = regular if not regular.empty else ranking
        priorities.insert(0, 'guardrail_deficit')
        ascending.insert(0, True)
    if 'calibration_log_loss' in best:
        priorities.append('calibration_log_loss')
        ascending.append(True)
    chosen = best.sort_values(priorities, ascending=ascending, kind='stable').iloc[0]
    result = chosen.astype(object).where(pd.notna(chosen), None).to_dict()
    result['selection_basis'] = 'quality_constrained' if chosen.feasible else 'quality_fallback'
    result['floors'] = floors
    result['guardrails_missed'] = [key for key, floor in floors.items() if chosen[key] < floor - 1e-12]
    return result
