"""Evaluate paired spatial decisions on the same held-out measurements."""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

IDENTITY = ['source_dataset', 'specimen_id']
METADATA = ['fold', *IDENTITY, 'x', 'y', 'label', 'baseline_prediction']
INTERVAL_METRICS = (
    'critical_error_rate', 'cavity_miss_rate', 'uncertain_miss_rate', 'false_alarm_rate',
    'macro_f1', 'balanced_accuracy', 'release_rate', 'recall_3',
)


def _divide(numerator, denominator, *, fill=np.nan):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, fill), where=denominator != 0)


def _confusion(truth, predicted):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.ndim != 1 or truth.shape != predicted.shape:
        raise ValueError('Expected two equally sized one-dimensional label arrays')
    if not np.isin(truth, range(4)).all() or not np.isin(predicted, range(4)).all():
        raise ValueError('Labels must belong to the four classes 0–3')
    indices = truth.astype(int, copy=False) * 4 + predicted.astype(int, copy=False)
    return np.bincount(indices, minlength=16).reshape(4, 4)


def _matrix_metrics(matrix):
    """Accept one confusion matrix or a stack of specimen-bootstrap matrices."""
    true_support = matrix.sum(axis=-1)
    predicted_support = matrix.sum(axis=-2)
    correct = np.diagonal(matrix, axis1=-2, axis2=-1)
    total = true_support.sum(axis=-1)
    precision = _divide(correct, predicted_support, fill=0.)
    recall = _divide(correct, true_support, fill=0.)
    f1 = _divide(2 * correct, true_support + predicted_support, fill=0.)
    cavity_misses = matrix[..., 1, 3]
    uncertain_misses = matrix[..., 2, 3]
    critical_errors = cavity_misses + uncertain_misses
    critical_support = true_support[..., 1] + true_support[..., 2]
    false_alarms = matrix[..., 3, 1] + matrix[..., 3, 2]
    release_count = predicted_support[..., 3]
    poor_releases = matrix[..., 0, 3]
    overall_unsafe = critical_errors + poor_releases

    result = {
        'n': total,
        'accuracy': _divide(correct.sum(axis=-1), total),
        'macro_f1': f1.mean(axis=-1),
        'balanced_accuracy': _divide(recall.sum(axis=-1), (true_support > 0).sum(axis=-1)),
        'critical_support': critical_support,
        'critical_errors': critical_errors,
        'critical_error_rate': _divide(critical_errors, critical_support),
        'cavity_misses': cavity_misses,
        'cavity_miss_rate': _divide(cavity_misses, true_support[..., 1]),
        'uncertain_misses': uncertain_misses,
        'uncertain_miss_rate': _divide(uncertain_misses, true_support[..., 2]),
        'false_alarms': false_alarms,
        'false_alarm_rate': _divide(false_alarms, true_support[..., 3]),
        'release_count': release_count,
        'release_rate': _divide(release_count, total),
        'true3_nonrelease_rate': _divide(true_support[..., 3] - correct[..., 3], true_support[..., 3]),
        'critical_given_release_rate': _divide(critical_errors, release_count),
        'poor_measurement_releases': poor_releases,
        'poor_measurement_release_rate': _divide(poor_releases, true_support[..., 0]),
        'overall_unsafe_clearances': overall_unsafe,
        'overall_unsafe_clearance_rate': _divide(overall_unsafe, total - true_support[..., 3]),
    }
    for label in range(4):
        result.update({
            f'precision_{label}': precision[..., label],
            f'recall_{label}': recall[..., label],
            f'f1_{label}': f1[..., label],
            f'support_{label}': true_support[..., label],
        })
    return result


def spatial_metrics(y_true, y_pred):
    """Match four_label_v2 error counts, with rates as fractions and absent rates as NaN.

    Macro F1 averages four fixed labels with zero for absent terms. Balanced
    accuracy averages recall over the true labels present in the evaluated slice.
    """
    return {name: value.item() for name, value in _matrix_metrics(_confusion(y_true, y_pred)).items()}


def _diagnostic_slices(frame):
    """Derive evaluation-only masks from annotated neighbors within each specimen."""
    labels = frame.label.to_numpy()
    has_neighbor = np.zeros(len(frame), dtype=bool)
    neighbor_count = np.zeros(len(frame), dtype=int)
    near_clear = np.zeros(len(frame), dtype=bool)
    near_risk = np.zeros(len(frame), dtype=bool)
    component_size = np.zeros(len(frame), dtype=int)
    for indices in frame.groupby(IDENTITY, sort=False).indices.values():
        coordinates = frame.iloc[indices][['x', 'y']].to_numpy(dtype=float)
        local_label = labels[indices]
        if (not np.isfinite(coordinates).all()
                or len(np.unique(coordinates, axis=0)) != len(coordinates)):
            raise ValueError('Diagnostics require finite, distinct coordinates within each specimen')

        pairs = np.empty((0, 2), dtype=int)
        if len(coordinates) > 1:
            tree = cKDTree(coordinates)
            spacing = np.median(tree.query(coordinates, k=2)[0][:, 1])
            # This descriptive stencil is fixed independently of the spatial search grid.
            pairs = tree.query_pairs(1.5 * spacing, output_type='ndarray')
        left, right = pairs.T
        this_point = np.r_[left, right]
        other_point = np.r_[right, left]
        neighbor_count[indices] = np.bincount(this_point, minlength=len(indices))
        has_neighbor[indices[this_point]] = True
        near_clear[indices[this_point[local_label[other_point] == 3]]] = True
        near_risk[indices[this_point[np.isin(local_label[other_point], [1, 2])]]] = True

        # Only directly connected class-1 annotations contribute to a cavity component.
        cavity_edges = (local_label[left] == 1) & (local_label[right] == 1)
        graph = coo_matrix(
            (np.ones(cavity_edges.sum()), (left[cavity_edges], right[cavity_edges])),
            shape=(len(indices), len(indices)),
        ).tocsr()
        _, components = connected_components(graph, directed=False)
        component_size[indices] = np.bincount(components)[components]

    cavity = labels == 1
    risk = np.isin(labels, [1, 2])
    clear = labels == 3
    return {
        'cavity_singleton_with_neighbors': cavity & (component_size == 1) & has_neighbor,
        'cavity_small_component_2_to_4': cavity & (component_size >= 2) & (component_size <= 4),
        'cavity_component_5_or_more': cavity & (component_size >= 5),
        'risk_adjacent_to_clear': risk & near_clear,
        'risk_with_neighbors_away_from_clear': risk & has_neighbor & ~near_clear,
        'clear_adjacent_to_risk': clear & near_risk,
        'clear_with_neighbors_away_from_risk': clear & has_neighbor & ~near_risk,
        'no_neighbors': ~has_neighbor,
        'sparse_neighborhood_1_or_2': has_neighbor & (neighbor_count <= 2),
    }


def _paired_metrics(frame):
    truth = frame.label.to_numpy()
    baseline = frame.baseline_prediction.to_numpy()
    prediction = frame.prediction.to_numpy()
    original, spatial = spatial_metrics(truth, baseline), spatial_metrics(truth, prediction)
    risk = np.isin(truth, [1, 2])
    return {
        **spatial,
        **{f'delta_{key}': spatial[key] - original[key] for key in INTERVAL_METRICS},
        'changed_predictions': int(np.count_nonzero(baseline != prediction)),
        'prevented_critical_errors': int(np.count_nonzero(risk & (baseline == 3) & (prediction != 3))),
        'introduced_critical_errors': int(np.count_nonzero(risk & (baseline != 3) & (prediction == 3))),
    }


def _bootstrap(frame, draws, seed):
    """Resample specimens with replacement, pairing baseline and spatial counts."""
    decisions = []
    references = []
    for _, rows in frame.groupby(IDENTITY, sort=True):
        decisions.append(_confusion(rows.label, rows.prediction).ravel())
        references.append(_confusion(rows.label, rows.baseline_prediction).ravel())
    n = len(decisions)
    if n < 2:
        return []
    weights = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=draws)
    new = _matrix_metrics((weights @ decisions).reshape(draws, 4, 4))
    old = _matrix_metrics((weights @ references).reshape(draws, 4, 4))
    observed = _paired_metrics(frame)
    output = []
    for metric in INTERVAL_METRICS:
        deltas = new[metric] - old[metric]
        finite = deltas[np.isfinite(deltas)]
        lower, upper = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
        output.append({
            'metric': metric, 'delta': observed[f'delta_{metric}'], 'lower_95': lower,
            'upper_95': upper, 'specimens': n, 'valid_draws': len(finite),
        })
    return output


def summarize(predictions, *, draws=2000, seed=42):
    """Return results by baseline, strategy, frozen fold, specimen and diagnostic slice.

    Each baseline requires an explicit untouched strategy='baseline' row per point.
    Bootstrap intervals describe these held-out decisions conditional on their fits.
    """
    if draws < 1:
        raise ValueError('At least one bootstrap draw is required')
    tables = {name: [] for name in ('pooled', 'fold', 'specimen', 'diagnostic',
                                   'paired_bootstrap', 'paired_pointwise')}
    for baseline, frame in predictions.groupby('baseline', sort=True):
        reference = frame[frame.strategy == 'baseline'].sort_values('instance_id').reset_index(drop=True)
        if (reference.empty or reference.instance_id.duplicated().any()
                or not reference.prediction.equals(reference.baseline_prediction)):
            raise ValueError(f'{baseline}: expected one untouched reference row per measurement')
        masks = _diagnostic_slices(reference)
        arms = {}
        for strategy, rows in frame.groupby('strategy', sort=True):
            rows = rows.sort_values('instance_id').reset_index(drop=True)
            keys = [*METADATA, 'instance_id']
            if not reference[keys].equals(rows[keys]):
                raise ValueError(f'{baseline}/{strategy}: measurements differ from the reference')
            arms[strategy] = rows
            method = {'baseline': baseline, 'strategy': strategy}
            tables['pooled'].append({**method, **_paired_metrics(rows)})
            for fold, subset in rows.groupby('fold', sort=True):
                tables['fold'].append({**method, 'fold': fold, **_paired_metrics(subset)})
            for (source, specimen), subset in rows.groupby(IDENTITY, sort=True):
                tables['specimen'].append({
                    **method, 'source_dataset': source, 'specimen_id': specimen,
                    'fold': subset.fold.iloc[0], **_paired_metrics(subset),
                })
            for name, mask in masks.items():
                if mask.any():
                    tables['diagnostic'].append({**method, 'slice': name, **_paired_metrics(rows[mask])})
            if strategy != 'baseline':
                tables['paired_bootstrap'].extend({**method, **row} for row in _bootstrap(rows, draws, seed))
        if 'pointwise' in arms:
            pointwise = arms['pointwise'].prediction
            for strategy, rows in arms.items():
                if strategy not in ('baseline', 'pointwise'):
                    paired = rows.assign(baseline_prediction=pointwise)
                    tables['paired_pointwise'].extend(
                        {'baseline': baseline, 'strategy': strategy, **row}
                        for row in _bootstrap(paired, draws, seed)
                    )
    return {name: pd.DataFrame.from_records(rows) for name, rows in tables.items()}
