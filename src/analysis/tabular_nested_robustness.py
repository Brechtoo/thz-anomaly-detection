"""Quantify paired specimen influence and uncertainty from frozen outer-fold decisions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.spatial_postprocessing_report import _matrix_metrics
from src.experiments.tabular_nested_hpo import OUTPUT, SEED

METRICS = ('critical_error_rate', 'false_alarm_rate', 'macro_f1', 'balanced_accuracy')
RATE_SCALE = {'critical_error_rate': 100, 'false_alarm_rate': 100}


def specimen_sensitivity(new, svm, folds, *, draws=2000, seed=SEED):
    """Rows are paired specimen matrices in the same fold/specimen order as the HPO report."""
    if (new.shape != svm.shape or new.shape != (len(folds), 4, 4)
            or np.any(new < 0) or np.any(svm < 0) or not np.array_equal(new.sum(2), svm.sum(2))):
        raise ValueError('Invalid paired confusion matrices or mismatched true class counts')
    rng = np.random.default_rng(seed)
    weights = np.zeros((draws, len(folds)), dtype=int)
    for fold in np.unique(folds):
        positions = np.flatnonzero(folds == fold)
        weights[:, positions] = rng.multinomial(len(positions), np.full(len(positions), 1 / len(positions)), draws)

    def estimates(matrices):
        total = matrices.sum(axis=0)
        boot = (weights @ matrices.reshape(len(folds), 16)).reshape(-1, 4, 4)
        return _matrix_metrics(total), _matrix_metrics(boot), _matrix_metrics(total - matrices)

    point, boot, without = estimates(new)
    svm_point, svm_boot, svm_without = estimates(svm)
    intervals, influence = [], {}
    for name in METRICS:
        scale = RATE_SCALE.get(name, 1)
        lo, hi = np.nanquantile(boot[name] * scale, [.025, .975])
        svm_lo, svm_hi = np.nanquantile(svm_boot[name] * scale, [.025, .975])
        delta_lo, delta_hi = np.nanquantile((boot[name] - svm_boot[name]) * scale, [.025, .975])
        intervals.append({
            'metric': name, 'model_estimate': point[name] * scale, 'model_lower_95': lo, 'model_upper_95': hi,
            'svm_estimate': svm_point[name] * scale, 'svm_lower_95': svm_lo, 'svm_upper_95': svm_hi,
            'delta': (point[name] - svm_point[name]) * scale,
            'delta_lower_95': delta_lo, 'delta_upper_95': delta_hi, 'draws': draws, 'seed': seed,
        })
        influence[f'delta_without_{name}'] = (without[name] - svm_without[name]) * scale
    influence['critical_support'] = new[:, 1:3].sum(axis=(1, 2))
    influence['critical_errors'] = new[:, 1:3, 3].sum(axis=1)
    influence['svm_critical_errors'] = svm[:, 1:3, 3].sum(axis=1)
    influence['delta_errors_on_specimen'] = influence['critical_errors'] - influence['svm_critical_errors']
    return pd.DataFrame(intervals), pd.DataFrame(influence)


def write_analysis(output):
    report = output / 'report'
    matrices = pd.read_csv(report / 'confusion.csv').query(
        "scope == 'specimen_id' and complete and mode in ('regular', 'adapted')"
    )
    cells = pd.MultiIndex.from_product([range(4), range(4)], names=['true_label', 'predicted_label'])

    def pivot(block):
        table = block.pivot(index=['fold', 'group'], columns=['true_label', 'predicted_label'], values='count')
        table = table.reindex(columns=cells).sort_index()
        if table.isna().any(axis=None) or not table.index.get_level_values('group').is_unique:
            raise ValueError('Missing cell or inconsistent specimen identity')
        return table

    groups = matrices.groupby(['model', 'features', 'mode'], sort=True)
    intervals, influence = [], []
    for (model, features, mode), block in groups:
        if model == 'rbf_svm':
            continue
        new = pivot(block)
        baseline = pivot(groups.get_group(('rbf_svm', features, mode)))
        pd.testing.assert_index_equal(new.index, baseline.index)
        metadata = new.index.to_frame(index=False).rename(columns={'group': 'specimen_id'})
        meta = {'model': model, 'features': features, 'mode': mode}
        bounds, bodies = specimen_sensitivity(new.to_numpy().reshape(-1, 4, 4),
                                              baseline.to_numpy().reshape(-1, 4, 4), metadata.fold.to_numpy())
        intervals.append(bounds.assign(**meta))
        influence.append(pd.concat([metadata, bodies], axis=1).assign(**meta))

    bounds = pd.concat(intervals, ignore_index=True)
    keys = ['model', 'features', 'mode']
    check = bounds.loc[bounds.metric.eq('critical_error_rate')].merge(
        pd.read_csv(report / 'paired_critical_intervals.csv'), on=keys, validate='one_to_one'
    )
    if len(check) != len(intervals):
        raise ValueError('Missing matching comparison in the HPO report')
    np.testing.assert_allclose(check[['delta_lower_95', 'delta_upper_95']], check[['lower_95', 'upper_95']], atol=1e-10)
    folder = output / 'robustness'
    folder.mkdir(exist_ok=True)
    for name, frame in [('paired_intervals', bounds), ('leave_one_specimen_out', pd.concat(influence))]:
        temp = folder / f'.{name}.csv'
        frame.to_csv(temp, index=False)
        temp.replace(folder / f'{name}.csv')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, default=OUTPUT)
    write_analysis(cli.parse_args().output)
