"""Explore class-3 release rules on the frozen four-class HeRAI SVM OOF predictions."""

import argparse
import importlib.metadata
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlparse

import matplotlib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.data.specimen_splits import REFERENCES
from src.data.split_manifest import sha256_file
from src.evaluation.metrics import (
    ERROR_METRICS_VERSION,
    calculate_error_counts,
    critical_breakdown,
)
from src.experiments.feature_grid import KEYS, LABELS, probabilities_in_label_order
from src.experiments.tracking import PROJECT_TRACKING_URI, set_project_experiment
from src.features.specimen_inputs import (
    ANCHOR,
    FOLDS,
    SPLITS,
    read_json,
    require_hash,
    write_json,
)
from src.paths import ANALYSIS_RESULTS, EXPERIMENTS_ROOT, PROJECT_ROOT

MODELS = {
    'manual+tsfel': 'Manuell + TSFEL',
    'manual': 'Manuell',
    'manual+tsfel+physical_v2': 'Manuell + TSFEL + Physik V2',
}
MODEL = 'rbf_svm'
EXPERIMENT = 'thz-rbf-svm-oof-thresholds'
TARGET_PERCENT = 2
PROBABILITIES = [f'p_{label}' for label in LABELS]
STRATEGIES = {'review': 'Prüfen', 'class2': 'Klasse 2', 'argmax012': 'Bestes p₀–p₂'}
GATES = ('predict', 'argmax')
THRESHOLDS = np.arange(101) / 100
COLORS = dict(zip(MODELS, ('#146c61', '#435778', '#aa572b'), strict=True))
POINTS = {
    'baseline': 'Ohne Schwelle',
    'pooled_lt2': 'Gepoolt unter 2 %',
    'all_folds_lt2': 'Alle Folds unter 2 %',
}


def frozen_train(root):
    """Verify the training-only cohort against its split, group and historical mirrors."""
    manifest = read_json(FOLDS / 'manifest.json')
    train_path = SPLITS / 'train.csv'
    folds_path = FOLDS / 'specimen_folds.csv'
    historic = ANCHOR / 'cohort_folds.csv'
    require_hash(train_path, manifest['train_split_sha256'])
    require_hash(folds_path, manifest['files'][folds_path.name])
    require_hash(historic, manifest['source_sha256'])
    train = pd.read_parquet(root / 'cohort.parquet')
    train = train.sort_values('instance_id', ignore_index=True)
    folds = pd.read_csv(folds_path).set_index('specimen_id')['fold']
    if not folds.index.is_unique:
        raise ValueError('Specimen fold assignment contains duplicates')
    pd.testing.assert_frame_equal(
        train[['instance_id', 'label']], pd.read_csv(train_path).sort_values('instance_id', ignore_index=True),
    )
    pd.testing.assert_frame_equal(
        train[KEYS], pd.read_csv(historic, usecols=KEYS).sort_values('instance_id', ignore_index=True)[KEYS],
    )
    pd.testing.assert_series_equal(train.specimen_id.map(folds).rename('fold'), train.fold)
    expected_counts = {int(fold): count for fold, count in manifest['points_per_fold'].items()}
    if (not train.split.eq('train').all() or not train.instance_id.is_unique
            or train[KEYS].isna().any().any() or train.specimen_id.isin(REFERENCES).any()
            or set(train.label) != set(LABELS) or set(train.specimen_id) != set(folds.index)
            or train.groupby('fold').size().to_dict() != expected_counts):
        raise ValueError('Unexpected frozen training cohort')
    return train


def verified_oof(root, train, client, parent):
    plan = read_json(root / 'plan.json')
    if plan['classes'] != LABELS.tolist() or plan['error_metrics'] != ERROR_METRICS_VERSION:
        raise ValueError('Unexpected class order or HeRAI error definitions in the Grid plan')
    for filename in ('src/experiments/feature_grid.py', 'src/experiments/training.py'):
        require_hash(PROJECT_ROOT / filename, plan['source_sha256'][filename])
    if parent.info.status != 'FINISHED':
        raise ValueError('Expected completed Grid parent')
    parameters = plan['models'][MODEL]
    input_runs, result = [], {}
    for model in MODELS:
        parts = []
        for fold in sorted(train.fold.unique()):
            stage = f'fold_{fold}'
            folder = root / 'fits' / model / MODEL / stage
            metrics = read_json(folder / 'metrics.json')
            path = folder / 'predictions.parquet'
            digest = metrics['predictions_sha256']
            require_hash(path, digest)
            run = client.get_run(metrics['run_id'])
            tags = run.data.tags
            if (run.info.status != 'FINISHED' or run.info.experiment_id != parent.info.experiment_id
                    or tags.get('mlflow.parentRunId') != parent.info.run_id or tags.get('stage') != stage
                    or tags.get('feature_configuration') != model or tags.get('model') != MODEL
                    or metrics['reused'] or dict(run.data.params) != parameters):
                raise ValueError(f'Unexpected MLflow provenance for {model}, {stage}')
            uri = urlparse(run.info.artifact_uri)
            if uri.scheme != 'file':
                raise ValueError('Expected the existing local MLflow artifact store')
            artifact = Path(unquote(uri.path))
            require_hash(artifact / path.name, digest)
            require_hash(artifact / 'metrics.json', sha256_file(folder / 'metrics.json'))
            frame = pd.read_parquet(path).sort_values('instance_id', ignore_index=True)
            if list(frame.columns) != [*KEYS, 'prediction', *PROBABILITIES] or not frame.prediction.isin(LABELS).all():
                raise ValueError(f'Unexpected OOF prediction schema in {path}')
            probabilities_in_label_order(frame[PROBABILITIES].to_numpy(), plan['classes'])
            pd.testing.assert_frame_equal(
                frame[KEYS], train.loc[train.fold.eq(fold), KEYS].reset_index(drop=True),
            )
            input_runs.append({'features': model, 'fold': int(fold), 'run_id': run.info.run_id,
                               'rows': len(frame), 'predictions_sha256': digest})
            parts.append(frame)
        result[model] = pd.concat(parts, ignore_index=True).sort_values('instance_id', ignore_index=True)
        pd.testing.assert_frame_equal(result[model][KEYS], train[KEYS])
    return result, input_runs, parameters


def route(base, probabilities, threshold, strategy):
    """Keep base decisions except class 3 below the inclusive release threshold."""
    rejected = (base == 3) & (probabilities[:, 3] < threshold)
    prediction = base.copy()
    if strategy == 'argmax012':
        prediction[rejected] = probabilities[rejected, :3].argmax(axis=1)
    else:
        prediction[rejected] = {'review': -1, 'class2': 2}[strategy]
    return prediction


def threshold_below_target(y, base, p3):
    """Lowest floating-point threshold at which the critical rate is strictly below 2%."""
    critical = np.isin(y, [1, 2])
    support = int(critical.sum())
    if support == 0:
        raise ValueError('The critical target needs at least one true label 1/2')
    allowed = (TARGET_PERCENT * support - 1) // 100
    scores = p3[critical & (base == 3)]
    if len(scores) <= allowed:
        return 0.0
    boundary_index = len(scores) - allowed - 1
    boundary = np.partition(scores, boundary_index)[boundary_index]
    return float(np.nextafter(boundary, np.inf))


def rate(count, denominator):
    return 100 * count / denominator if denominator else np.nan


def decision_metrics(y, prediction, hard):
    counts = calculate_error_counts(y, prediction)
    counts.update(critical_breakdown(y, prediction))
    true_critical = np.isin(y, [1, 2])
    true_3 = y == 3
    released = prediction == 3
    review = prediction == -1
    added = released & (hard != 3)
    critical_support, class3_support = int(true_critical.sum()), int(true_3.sum())
    if not critical_support:
        counts['critical_error_rate'] = np.nan
    if not class3_support:
        counts['false_alarm_rate'] = np.nan
    return {
        'n': len(y), 'critical_support': critical_support, 'class3_support': class3_support,
        **counts,
        'cleared': int(released.sum()), 'clear_rate': rate(released.sum(), len(y)),
        'nonclear_rate': rate((~released).sum(), len(y)),
        'review': int(review.sum()), 'review_rate': rate(review.sum(), len(y)),
        'true3_review': int((true_3 & review).sum()),
        'true3_review_rate': rate((true_3 & review).sum(), class3_support),
        'true3_nonclear': int((true_3 & ~released).sum()),
        'true3_nonclear_rate': rate((true_3 & ~released).sum(), class3_support),
        'critical_among_cleared_rate': rate(counts['critical_errors'], released.sum()),
        'removed_vs_predict': int(((hard == 3) & ~released).sum()),
        'newly_cleared_vs_predict': int(added.sum()),
        'new_critical_vs_predict': int((added & true_critical).sum()),
    }


def scopes(frame):
    yield 'pooled', 'all', np.arange(len(frame))
    for column in ('fold', 'specimen_id'):
        for key, positions in frame.groupby(column).indices.items():
            yield column, str(key), positions


def decision_tables(frames):
    sweeps, points, by_class = [], [], []
    for model, frame in frames.items():
        y, hard = frame.label.to_numpy(), frame.prediction.to_numpy()
        probabilities = frame[PROBABILITIES].to_numpy()
        argmax = probabilities.argmax(axis=1)
        groups = list(scopes(frame))
        for gate, base in zip(GATES, (hard, argmax), strict=True):
            pooled = threshold_below_target(y, base, probabilities[:, 3])
            fold_targets = [
                threshold_below_target(y[idx], base[idx], probabilities[idx, 3])
                for scope, _, idx in groups if scope == 'fold'
            ]
            operating = dict(zip(POINTS, [0.0, pooled, max(fold_targets)], strict=True))
            for strategy in STRATEGIES:
                keys = {'features': model, 'gate': gate, 'routing': strategy}
                for threshold in THRESHOLDS:
                    prediction = route(base, probabilities, threshold, strategy)
                    sweeps.append({**keys, 'threshold': threshold, **decision_metrics(y, prediction, hard)})
                for point, threshold in operating.items():
                    prediction = route(base, probabilities, threshold, strategy)
                    for scope, group, idx in groups:
                        location = {**keys, 'point': point, 'threshold': threshold, 'scope': scope, 'group': group}
                        points.append({**location, **decision_metrics(y[idx], prediction[idx], hard[idx])})
                        for label in LABELS:
                            mask = idx[y[idx] == label]
                            counts = np.bincount(prediction[mask] + 1, minlength=5)
                            by_class.append({**location, 'true_label': label, 'support': len(mask),
                                             **dict(zip(('pred_review', 'pred_0', 'pred_1', 'pred_2', 'pred_3'),
                                                        counts, strict=True))})
    return tuple(pd.DataFrame(rows) for rows in (sweeps, points, by_class))


def reliability(score, event):
    bins = np.minimum((score * 10).astype(int), 9)
    rows = []
    for bin_id in np.unique(bins):
        mask = bins == bin_id
        rows.append({'bin': int(bin_id), 'n': int(mask.sum()), 'mean_score': score[mask].mean(),
                     'event_rate': event[mask].mean()})
    ece = sum(row['n'] * abs(row['mean_score'] - row['event_rate']) for row in rows) / len(score)
    return rows, ece


def calibration_tables(frames):
    quality, bins, distribution = [], [], []
    for model, frame in frames.items():
        for scope, group, idx in scopes(frame):
            if scope == 'specimen_id':
                continue
            probabilities = frame[PROBABILITIES].to_numpy()[idx]
            y, hard = frame.label.to_numpy()[idx], frame.prediction.to_numpy()[idx]
            truth = y[:, None] == LABELS
            argmax = probabilities.argmax(axis=1)
            row = {'features': model, 'scope': scope, 'group': group, 'n': len(y),
                   'log_loss': log_loss(y, probabilities, labels=LABELS),
                   'brier_multiclass': np.mean(np.sum((probabilities - truth) ** 2, axis=1)),
                   'argmax_diff': int((argmax != hard).sum()),
                   'argmax_class3_diff': int(((argmax == 3) != (hard == 3)).sum())}
            green = hard == 3
            checks = {f'p_{label}': (probabilities[:, label], truth[:, label]) for label in LABELS}
            checks['p_1+p_2|predict=3'] = (probabilities[green, 1:3].sum(axis=1), np.isin(y[green], [1, 2]))
            for name, (score, event) in checks.items():
                values, ece = reliability(score, event)
                row[f'ece_{name}'] = ece
                row[f'brier_{name}'] = np.mean((score - event) ** 2)
                row[f'mean_{name}'] = score.mean()
                row[f'frequency_{name}'] = event.mean()
                bins.extend({'features': model, 'scope': scope, 'group': group, 'score': name, **value}
                            for value in values)
            quality.append(row)
            for label in LABELS:
                for subset, mask in [('all', y == label), ('predict=3', (y == label) & green)]:
                    scores = probabilities[mask, 3]
                    quantiles = np.quantile(scores, [0.05, 0.25, 0.50, 0.75, 0.95]) if len(scores) else [np.nan] * 5
                    distribution.append({'features': model, 'scope': scope, 'group': group, 'true_label': label,
                                         'subset': subset, 'n': len(scores),
                                         **dict(zip(('q05', 'q25', 'q50', 'q75', 'q95'), quantiles, strict=True))})
    return tuple(pd.DataFrame(rows) for rows in (quality, bins, distribution))


def plot_tradeoffs(sweep, operating, output):
    strategies = {'review': '-', 'class2': ':', 'argmax012': '--'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    primary = sweep.loc[sweep.gate.eq('predict')]
    for (model, strategy), rows in primary.groupby(['features', 'routing'], sort=False):
        color = COLORS[model]
        axes[0, 0].plot(rows.false_alarm_rate, rows.critical_error_rate, strategies[strategy], color=color)
        if strategy == 'review':
            axes[0, 1].plot(rows.clear_rate, rows.critical_error_rate, color=color)
            axes[1, 0].plot(rows.true3_review_rate, rows.critical_error_rate, color=color)
            alternative = sweep.loc[sweep.gate.eq('argmax') & sweep.features.eq(model) & sweep.routing.eq('review')]
            axes[0, 1].plot(alternative.clear_rate, alternative.critical_error_rate, '--', color=color, alpha=.8)
            chosen = operating.loc[operating.features.eq(model) & operating.gate.eq('predict')
                                   & operating.routing.eq('review') & operating.scope.eq('pooled')
                                   & operating.point.eq('pooled_lt2')].iloc[0]
            axes[0, 1].scatter(chosen.clear_rate, chosen.critical_error_rate, color=color, s=30, zorder=4)
            axes[1, 1].plot(rows.threshold, rows.critical_1_rate, color=color)
            axes[1, 1].plot(rows.threshold, rows.critical_2_rate, '--', color=color)
    for axis, label in zip(axes.flat, [
        'Fehlalarmrate nach HeRAI (%)', 'Freigabe aller Messungen (%)',
        'Prüfentscheidung bei wahrer Klasse 3 (%)', 'Schwelle für p₃',
    ], strict=True):
        axis.set(xlabel=label, ylabel='Kritische Fehlerrate (%)')
        axis.axhline(TARGET_PERCENT, color='0.4', alpha=.55, linewidth=1)
        axis.grid(alpha=.2)
    axes[0, 0].legend(handles=[Line2D([0], [0], color='.3', linestyle=linestyle, label=STRATEGIES[key])
                              for key, linestyle in strategies.items()], fontsize=8)
    axes[0, 1].legend(handles=[Line2D([0], [0], color='.3', label='SVC predict'),
                              Line2D([0], [0], color='.3', ls='--', label='Wahrscheinlichkeits-Argmax')],
                     fontsize=8)
    axes[1, 0].legend(handles=[Line2D([0], [0], color=COLORS[key], label=name) for key, name in MODELS.items()],
                     fontsize=8)
    axes[1, 1].legend(handles=[Line2D([0], [0], color='.3', label='1 → 3'),
                              Line2D([0], [0], color='.3', ls='--', label='2 → 3')], fontsize=8)
    fig.savefig(output / 'tradeoffs.png', dpi=170)
    plt.close(fig)


def plot_calibration(frames, bins, output):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), layout='constrained')
    for i, (model, frame) in enumerate(frames.items()):
        for column, score in enumerate(('p_3', 'p_1+p_2|predict=3')):
            axis = axes[i, column]
            rows = bins.loc[bins.features.eq(model) & bins.scope.eq('pooled') & bins.score.eq(score)]
            axis.plot([0, 1], [0, 1], color='.6', lw=1)
            axis.plot(rows.mean_score, rows.event_rate, lw=1, color=COLORS[model])
            axis.scatter(rows.mean_score, rows.event_rate, s=14 + 75 * rows.n / rows.n.max(), color=COLORS[model])
            axis.set(xlim=(0, 1), ylim=(0, 1), ylabel='Beobachteter Anteil',
                     xlabel='Mittleres p₃' if column == 0 else 'Mittleres p₁ + p₂ bei predict = 3')
        axis = axes[i, 2]
        for label, color in zip(LABELS, ['#747782', '#ab5354', '#986c1d', '#076c5e'], strict=True):
            values = frame.loc[frame.label.eq(label), 'p_3']
            axis.hist(values, bins=np.linspace(0, 1, 21), weights=np.full(len(values), 1 / len(values)),
                      histtype='step', color=color, lw=1.5, label=f'Klasse {label}')
        axis.set(xlim=(0, 1), xlabel='p₃', ylabel='Anteil innerhalb der wahren Klasse')
        axes[i, 0].set_title(MODELS[model], loc='left', fontsize=11)
        axis.legend(fontsize=8)
    fig.savefig(output / 'wahrscheinlichkeiten.png', dpi=170)
    plt.close(fig)


def analyze(root, output, log=True):
    if output.is_relative_to(root) or output.exists():
        raise ValueError(f'Use a new output directory outside the source run: {output}')
    mlflow.set_tracking_uri(PROJECT_TRACKING_URI)
    client = mlflow.MlflowClient()
    parent = client.get_run(read_json(root / 'run.json')['run_id'])
    train = frozen_train(root)
    frames, runs, parameters = verified_oof(root, train, client, parent)
    sweeps, points, by_class = decision_tables(frames)
    quality, bins, distribution = calibration_tables(frames)
    manifest = {
        'scope': 'exploratory training OOF only', 'source_parent_run_id': parent.info.run_id,
        'source_experiment_id': parent.info.experiment_id, 'source': str(root), 'folds': runs,
        'cohort_sha256': sha256_file(root / 'cohort.parquet'), 'plan_sha256': sha256_file(root / 'plan.json'),
        'script_sha256': sha256_file(__file__), 'rows': len(train), 'specimens': train.specimen_id.nunique(),
        'model_params': parameters, 'classes': LABELS.tolist(), 'error_metrics': ERROR_METRICS_VERSION,
        'release_if': 'gate predicts 3 and p_3 >= threshold', 'gates': list(GATES),
        'routings': STRATEGIES, 'review_code': -1, 'threshold_grid': THRESHOLDS.tolist(),
        'strict_critical_target_percent': TARGET_PERCENT, 'operating_points': POINTS,
        'calibration': '10 equal-width bins, occupancy-weighted ECE; no calibrator fitted here',
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name)
                     for name in ('numpy', 'pandas', 'pyarrow', 'scikit-learn', 'matplotlib', 'mlflow')},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.thresholds-') as scratch:
        staging = Path(scratch) / 'analysis'
        staging.mkdir()
        for name, table in {
            'sweep': sweeps, 'operating_points': points, 'outcomes_by_true_class': by_class,
            'calibration': quality, 'reliability_bins': bins, 'p3_quantiles': distribution,
        }.items():
            table.to_csv(staging / f'{name}.csv', index=False)
        plot_tradeoffs(sweeps, points, staging)
        plot_calibration(frames, bins, staging)
        manifest['outputs_sha256'] = {path.name: sha256_file(path) for path in sorted(staging.iterdir())}
        write_json(staging / 'manifest.json', manifest)
        staging.rename(output)
    if log:
        set_project_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=output.name) as run:
            mlflow.set_tags({'source_grid_run_id': parent.info.run_id, 'scope': 'exploratory_training_oof'})
            mlflow.log_params({'candidate_count': len(frames), 'fold_count': train.fold.nunique(),
                               'critical_target_percent_strict': TARGET_PERCENT, 'threshold_step': 0.01})
            target = points.query("scope == 'pooled' and gate == 'predict' and routing == 'review' "
                                  "and point == 'pooled_lt2'")
            for row in target.itertuples(index=False):
                model_key = row.features.replace('+', '_')
                mlflow.log_metrics({f'{model_key}.{key}': getattr(row, key)
                                    for key in ('threshold', 'critical_error_rate', 'false_alarm_rate',
                                                'clear_rate', 'review_rate', 'true3_nonclear_rate')})
            mlflow.log_artifacts(str(output))
            mlflow.log_artifact(__file__, 'source')
            write_json(output / 'mlflow_run.json', {'run_id': run.info.run_id, 'experiment_id': run.info.experiment_id})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=EXPERIMENTS_ROOT / 'specimen_feature_grid_v1')
    parser.add_argument('--output', type=Path, default=ANALYSIS_RESULTS / 'rbf_svm_oof_thresholds_v1')
    parser.add_argument('--no-log', action='store_true', help='Regenerate artifacts without creating an MLflow run.')
    args = parser.parse_args()
    analyze(args.source.resolve(), args.output.resolve(), log=not args.no_log)


if __name__ == '__main__':
    main()
