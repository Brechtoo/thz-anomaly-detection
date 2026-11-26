"""Pair outer decisions to archived SVM decisions on identical measurements."""

import numpy as np
import pandas as pd

from src.evaluation.svm_nested_selection import decision_metrics
from src.experiments.svm_nested_hpo import KEYS, SEED, atomic_json

MODES = ('regular', 'hard', 'adapted')
METRICS = ('critical_error_rate', 'critical_1_rate', 'critical_2_rate', 'false_alarm_rate',
           'macro_f1', 'balanced_accuracy', 'release_rate')


def confusion(frame, mode):
    labels = frame.label.to_numpy(dtype=int)
    decisions = frame[f'prediction_{mode}'].to_numpy(dtype=int)
    if not np.isin(labels, range(4)).all() or not np.isin(decisions, range(4)).all():
        raise ValueError('Expected four-class decisions')
    return np.bincount(4 * labels + decisions, minlength=16).reshape(4, 4)


def bootstrap(new, old, mode, draws=2000):
    """Keep predictions paired and resample whole specimens within each outer fold."""
    rng = np.random.default_rng(SEED)
    new_totals = np.zeros((draws, 4, 4), dtype=int)
    old_totals = new_totals.copy()
    for fold, block in new.groupby('fold', sort=True):
        specimens = [(body, confusion(group, mode)) for body, group in block.groupby('specimen_id', sort=True)]
        path = old.loc[old.fold.eq(fold)]
        baseline = [confusion(path.loc[path.specimen_id.eq(body)], mode) for body, _ in specimens]
        weights = rng.multinomial(len(specimens), np.full(len(specimens), 1 / len(specimens)), size=draws)
        new_totals += (weights @ np.asarray([matrix.ravel() for _, matrix in specimens])).reshape(draws, 4, 4)
        old_totals += (weights @ np.asarray(baseline).reshape(len(baseline), 16)).reshape(draws, 4, 4)

    def critical(matrix):
        counts = matrix[:, 1:3, 3].sum(axis=1) * 100
        totals = matrix[:, 1:3].sum(axis=(1, 2))
        return np.divide(counts, totals, out=np.full(draws, np.nan), where=totals > 0)

    return np.nanquantile(critical(new_totals) - critical(old_totals), [.025, .975]).tolist()


def summaries(frame, model, features, modes):
    scopes = [('pooled', 'all', frame)]
    scopes.extend((key, str(group), rows) for key in ('fold', 'specimen_id')
                  for group, rows in frame.groupby(key, sort=True))
    metrics, matrices = [], []
    for scope, group, rows in scopes:
        for mode in modes:
            identity = {'model': model, 'features': features, 'mode': mode, 'scope': scope, 'group': group,
                        'fold': int(rows.fold.iloc[0]) if scope != 'pooled' else None,
                        'complete': frame.fold.nunique() == 3}
            values = decision_metrics(rows.label.to_numpy(), rows[f'prediction_{mode}'].to_numpy())
            metrics.append({**identity, **values})
            matrix = confusion(rows, mode)
            matrices.extend({**identity, 'true_label': label, 'predicted_label': prediction,
                             'count': matrix[label, prediction]}
                            for label in range(4) for prediction in range(4))
    return metrics, matrices


def write_report(output, evaluated, references, *, completed):
    report = output / 'report'
    report.mkdir(exist_ok=True)
    metrics, matrices, comparisons, intervals = [], [], [], []
    models = sorted({(model, features) for model, features, _ in evaluated})
    for features in sorted({features for features, _ in references}):
        reference = pd.concat([part for (combo, _), part in references.items() if combo == features],
                              ignore_index=True)
        records, confusions = summaries(reference, 'rbf_svm', features, ('regular', 'adapted'))
        metrics.extend(records)
        matrices.extend(confusions)
    for model, features in models:
        folds = sorted(fold for name, combo, fold in evaluated if (name, combo) == (model, features))
        new = pd.concat([evaluated[model, features, fold] for fold in folds], ignore_index=True)
        old = pd.concat([references[features, fold] for fold in folds], ignore_index=True)
        pd.testing.assert_frame_equal(new[KEYS], old[KEYS])
        records, confusions = summaries(new, model, features, MODES)
        metrics.extend(records)
        matrices.extend(confusions)
        baseline, _ = summaries(old, 'rbf_svm', features, ('regular', 'adapted'))
        controls = {(row['scope'], row['group'], row['mode']): row for row in baseline}
        for row in records:
            if row['mode'] == 'hard':
                continue
            control = controls[row['scope'], row['group'], row['mode']]
            comparisons.append({
                **{key: row[key] for key in ('model', 'features', 'mode', 'scope', 'group', 'fold', 'complete')},
                'critical_support': row['critical_support'], 'critical_errors': row['critical_errors'],
                'svm_critical_errors': control['critical_errors'],
                **{f'delta_{key}': row[key] - control[key] for key in METRICS},
            })
        for mode in ('regular', 'adapted'):
            lo, hi = bootstrap(new, old, mode)
            intervals.append({'model': model, 'features': features, 'mode': mode, 'folds': len(folds),
                              'lower_95': lo, 'upper_95': hi, 'draws': 2000, 'seed': SEED})
    tables = {
        'decision_metrics': pd.DataFrame(metrics), 'confusion': pd.DataFrame(matrices),
        'paired_svm': pd.DataFrame(comparisons), 'paired_critical_intervals': pd.DataFrame(intervals),
    }
    for name, table in tables.items():
        temp = report / f'.{name}.csv'
        table.to_csv(temp, index=False)
        temp.replace(report / f'{name}.csv')
    atomic_json(report / 'status.json', {'complete': completed, 'expected_outer_assessments': 45,
                                         'complete_outer_assessments': len(evaluated)})
    return tables['decision_metrics']
