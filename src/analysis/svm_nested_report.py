"""Report nested SVM outer-fold checkpoints without selecting on outer results."""

import argparse
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix, log_loss

from src.analysis.rbf_svm_oof_thresholds import frozen_train
from src.evaluation.metrics import calculate_error_counts, critical_breakdown
from src.evaluation.reporting import metrics_for_group
from src.features.specimen_inputs import read_json, require_hash
from src.paths import EXPERIMENTS_ROOT

META = ['instance_id', 'specimen_id', 'source_dataset', 'label', 'fold']
LABELS = [0, 1, 2, 3]
PROBA = [f'p_{label}' for label in LABELS]
REGULAR_PROBA = [f'regular_p_{label}' for label in LABELS]
METHODS = {
    'prediction_grid': 'Grid-Referenz',
    'prediction_regular': 'Regulär',
    'prediction_hard': 'Sicherheitsfit ohne Regel',
    'prediction_adapted': 'Adaptiert',
    'prediction_review': 'Explizite Prüfung',
}
FULL_SCOPE = ('pooled', 'all')
FEATURE_NAMES = {
    'manual': 'Manuell',
    'manual+tsfel': 'Manuell + TSFEL',
    'manual+tsfel+physical_v2': 'Manuell + TSFEL + Physik V2',
}
COLORS = ['#828282', '#365889', '#946039', '#18786d', '#82427a']
PAIRED_METRICS = (
    'critical_errors', 'critical_error_rate', 'critical_1_errors', 'critical_1_rate',
    'critical_2_errors', 'critical_2_rate', 'false_alarms', 'false_alarm_rate',
    'release_rate', 'macro_f1', 'balanced_accuracy', 'true3_nonrelease_rate',
)
PAIRS = (
    ('prediction_grid', 'prediction_regular'),
    ('prediction_grid', 'prediction_adapted'),
    ('prediction_regular', 'prediction_adapted'),
    ('prediction_hard', 'prediction_adapted'),
)
RESAMPLED_RATES = {
    'critical_error_rate': ('delta_critical_errors', 'critical_support'),
    'critical_1_rate': ('delta_critical_1_errors', 'critical_1_support'),
    'critical_2_rate': ('delta_critical_2_errors', 'critical_2_support'),
    'false_alarm_rate': ('delta_false_alarms', 'true3_support'),
}
RESAMPLING_DRAWS = 2000
RESAMPLING_SEED = 42


def rate(count, total):
    return 100 * count / total if total else np.nan


def decision_metrics(y, prediction, *, review=False):
    """Review has explicit coverage metrics and no four-class performance scores."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    metrics = calculate_error_counts(y, prediction)
    metrics.update(critical_breakdown(y, prediction))
    true3 = y == 3
    critical_support = int(np.isin(y, [1, 2]).sum())
    true3_support = int(true3.sum())
    released = prediction == 3
    abstained = prediction == -1
    if not review:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UndefinedMetricWarning)
            metrics.update(metrics_for_group(y, prediction))
    if not critical_support:
        metrics['critical_error_rate'] = np.nan
    if not true3_support:
        metrics['false_alarm_rate'] = np.nan
    metrics.update(
        points=len(y),
        critical_support=critical_support,
        true3_support=true3_support,
        released=int(released.sum()),
        release_rate=rate(released.sum(), len(y)),
        true3_nonrelease=int((true3 & ~released).sum()),
        true3_nonrelease_rate=rate((true3 & ~released).sum(), true3_support),
        review=int(abstained.sum()),
        review_rate=rate(abstained.sum(), len(y)),
        true3_review=int((true3 & abstained).sum()),
        true3_review_rate=rate((true3 & abstained).sum(), true3_support),
        critical_given_release=rate(metrics['critical_errors'], released.sum()),
    )
    return metrics


def probability_metrics(truth, probabilities):
    observed = np.eye(len(LABELS))[truth]
    error = probabilities - observed
    p3 = probabilities[:, 3]
    bins = np.minimum((p3 * 10).astype(int), 9)
    ece = sum(
        abs(error[mask, 3].mean()) * mask.sum()
        for index in range(10)
        if (mask := bins == index).any()
    ) / len(truth)
    return {
        'log_loss': log_loss(truth, probabilities, labels=LABELS),
        'multiclass_brier': np.square(error).sum(axis=1).mean(),
        'ece_p3': ece,
    }


def verify_probabilities(frame, columns):
    probabilities = frame[columns].to_numpy(dtype=float)
    if (not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError(f'Invalid probability channels {columns}')


def verified_frames(output, source, cohort):
    """Pair historical decisions to precisely the outer observations already written."""
    frames = {}
    for path in sorted((output / 'outer_oof').glob('*.parquet')):
        frame = pd.read_parquet(path).sort_values('instance_id', ignore_index=True)
        folds = frame.fold.unique().tolist()
        if not folds or not set(folds) <= {0, 1, 2}:
            raise ValueError(f'Unexpected outer folds in {path}')
        expected = cohort.loc[cohort.fold.isin(folds), META].reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[META], expected, check_dtype=False)
        for column in ('prediction_regular', 'prediction_hard', 'prediction_adapted'):
            if not frame[column].isin(LABELS).all():
                raise ValueError(f'Unexpected decision codes in {path} / {column}')
        if 'prediction_review' in frame and not frame.prediction_review.isin([-1, *LABELS]).all():
            raise ValueError(f'Unexpected review decision codes in {path}')
        verify_probabilities(frame, PROBA)
        verify_probabilities(frame, REGULAR_PROBA)
        historical_parts = []
        for fold in folds:
            directory = source / 'fits' / path.stem / 'rbf_svm' / f'fold_{fold}'
            baseline_path = directory / 'predictions.parquet'
            digest = read_json(directory / 'metrics.json')['predictions_sha256']
            require_hash(baseline_path, digest)
            historical_parts.append(pd.read_parquet(baseline_path, columns=[*META, 'prediction']))
        baseline = pd.concat(historical_parts, ignore_index=True).sort_values('instance_id', ignore_index=True)
        pd.testing.assert_frame_equal(baseline[META], expected, check_dtype=False)
        if not baseline.prediction.isin(LABELS).all():
            raise ValueError(f'Unexpected historical decision codes for {path.stem}')
        frames[path.stem] = frame.assign(prediction_grid=baseline.prediction)
    return frames


def scopes(frame):
    yield *FULL_SCOPE, frame
    for key in ('fold', 'specimen_id'):
        for identity, group in frame.groupby(key, sort=True):
            yield key, str(identity), group


def summarize(frames):
    metrics, classes, confusions, calibration, deltas = [], [], [], [], []
    for features, frame in frames.items():
        complete = frame.fold.nunique() == 3
        for scope, group, subset in scopes(frame):
            identity = {'features': features, 'scope': scope, 'group': group, 'complete': complete}
            local = {}
            y = subset.label.to_numpy(dtype=int)
            for column, name in METHODS.items():
                if column not in subset:
                    continue
                predictions = subset[column].to_numpy(dtype=int)
                is_review = column == 'prediction_review'
                values = decision_metrics(y, predictions, review=is_review)
                if scope == 'specimen_id':
                    values['fold'] = int(subset.fold.iloc[0])
                    values['source_dataset'] = subset.source_dataset.iloc[0]
                local[column] = values
                model_key = {**identity, 'method': name}
                metrics.append({**model_key, **values})
                decision_labels = [*LABELS, -1] if is_review else LABELS
                matrix = confusion_matrix(y, predictions, labels=decision_labels)
                confusions.extend({
                    **model_key, 'true_label': label, 'decision': decision, 'count': int(matrix[i, j]),
                } for i, label in enumerate(LABELS) for j, decision in enumerate(decision_labels))
                for label in LABELS:
                    support = int(matrix[label].sum())
                    released = int(matrix[label, 3])
                    reviewed = int(matrix[label, 4]) if is_review else 0
                    outcome = {
                        **model_key, 'true_label': label, 'support': support,
                        'released': released, 'release_rate': rate(released, support),
                        'reviewed': reviewed, 'review_rate': rate(reviewed, support),
                    }
                    if not is_review:
                        predicted = int(matrix[:, label].sum())
                        correct = int(matrix[label, label])
                        outcome.update(
                            precision=correct / predicted if predicted else np.nan,
                            recall=correct / support if support else np.nan,
                            f1=2 * correct / (support + predicted) if support + predicted else np.nan,
                        )
                    classes.append(outcome)
            if scope in ('pooled', 'fold'):
                for method, cols in [('Regulär', REGULAR_PROBA), ('Sicherheitsfit', PROBA)]:
                    calibration.append({
                        **identity, 'method': method, 'points': len(y),
                        **probability_metrics(y, subset[cols].to_numpy(dtype=float)),
                    })
            for before, after in PAIRS:
                deltas.append({
                    **identity, 'reference': METHODS[before], 'variant': METHODS[after],
                    'fold': local[before].get('fold'),
                    'critical_support': local[before]['critical_support'],
                    'critical_1_support': local[before]['critical_1_support'],
                    'critical_2_support': local[before]['critical_2_support'],
                    'true3_support': local[before]['true3_support'],
                    **{f'delta_{metric}': local[after][metric] - local[before][metric] for metric in PAIRED_METRICS},
                })
    return {
        'decision_metrics.csv': pd.DataFrame(metrics),
        'outcomes_by_true_class.csv': pd.DataFrame(classes),
        'confusion_matrices.csv': pd.DataFrame(confusions),
        'probability_metrics.csv': pd.DataFrame(calibration),
        'paired_deltas.csv': pd.DataFrame(deltas),
    }


def paired_specimen_intervals(deltas):
    """Resample paired specimen counts within outer folds with the fits held fixed."""
    rows = []
    keys = ['features', 'reference', 'variant']
    body = deltas.loc[deltas.scope.eq('specimen_id')].sort_values('group')
    columns = [column for pair in RESAMPLED_RATES.values() for column in pair]
    for identity, group in body.groupby(keys, sort=True):
        rng = np.random.default_rng(RESAMPLING_SEED)
        totals = np.zeros((RESAMPLING_DRAWS, len(columns)), dtype=int)
        for _, fold in group.groupby('fold', sort=True):
            weights = rng.multinomial(len(fold), np.full(len(fold), 1 / len(fold)), size=RESAMPLING_DRAWS)
            totals += weights @ fold[columns].to_numpy(dtype=int)
        for index, (metric, (errors, support)) in enumerate(RESAMPLED_RATES.items()):
            numerator, denominator = totals[:, index * 2:index * 2 + 2].T
            valid = denominator > 0
            samples = 100 * numerator[valid] / denominator[valid]
            low, high = np.percentile(samples, [2.5, 97.5]) if valid.any() else (np.nan, np.nan)
            rows.append({
                **dict(zip(keys, identity, strict=True)), 'metric': metric, 'complete': bool(group.complete.all()),
                'specimens': len(group), 'folds': group.fold.nunique(), 'support': int(group[support].sum()),
                'delta_pct_points': rate(group[errors].sum(), group[support].sum()),
                'p025_pct_points': low, 'p975_pct_points': high, 'valid_draws': int(valid.sum()),
                'draws': RESAMPLING_DRAWS, 'seed': RESAMPLING_SEED,
            })
    return pd.DataFrame(rows)


def search_progress(output, frames):
    plan_path = output / 'plan.json'
    if not plan_path.is_file():
        return pd.DataFrame()
    plan = read_json(plan_path)
    candidates = {item['candidate'] for item in plan['candidates']}
    rows = []
    for features in plan['features']:
        for outer in range(3):
            work = output / 'search' / features / f'outer_{outer}'
            checkpoints = []
            failures = list(work.glob('failure.json'))
            for inner in range(3):
                inner_dir = work / f'inner_{inner}'
                checkpoints.append(candidates.intersection(path.parent.name for path in inner_dir.glob('*/fit.json')))
                failures.extend(inner_dir.glob('failures/*.json'))
            rows.append({
                'features': features, 'outer_fold': outer, 'inner_fits_expected': 3 * len(candidates),
                'inner_fits_checkpointed': sum(map(len, checkpoints)), 'full_candidates': len(set.intersection(*checkpoints)),
                'outer_checkpointed': (output / 'outer' / features / f'fold_{outer}' / 'predictions.parquet').is_file(),
                'outer_reported': features in frames and frames[features].fold.eq(outer).any(),
                'persisted_failure_files': len(failures),
                'failure_paths': ' | '.join(str(path.relative_to(output)) for path in sorted(failures)),
            })
    return pd.DataFrame(rows)


def body_robustness(metrics):
    rows = []
    for (features, method), group in metrics.loc[metrics.scope.eq('specimen_id')].groupby(['features', 'method']):
        supported = group.loc[group.critical_support.gt(0)]
        if supported.empty:
            continue
        worst = supported.sort_values(['critical_error_rate', 'critical_support'], ascending=False).iloc[0]
        rows.append({
            'features': features, 'method': method, 'bodies': len(group), 'bodies_with_critical_support': len(supported),
            'bodies_critical_under_2pct': int(supported.critical_error_rate.lt(2).sum()),
            'body_mean_critical_rate': supported.critical_error_rate.mean(),
            'body_median_critical_rate': supported.critical_error_rate.median(),
            'body_max_critical_rate': worst.critical_error_rate, 'worst_body': worst.group,
            'worst_body_critical_errors': int(worst.critical_errors),
            'worst_body_critical_support': int(worst.critical_support),
        })
    return pd.DataFrame(rows)


def ordered_features(frames):
    return sorted(frames, key=lambda name: (list(FEATURE_NAMES).index(name) if name in FEATURE_NAMES else 99, name))


def render(output, frames, selections, tables, finalized):
    lines = [
        '# Äußere Gruppen-CV der verschachtelten RBF-SVM-Suche',
        '',
        'Die Feature-Sets wurden bereits anhand derselben OOF-Kohorte vorausgewählt. '
        'Die verschachtelte CV bewertet die innere Parameter- und Regelauswahl innerhalb dieser Vorauswahl. '
        'Die Ergebnisse bleiben explorativ. Validation und Test gehen hier nicht ein.',
        '',
    ]
    if (not finalized or set(FEATURE_NAMES) - set(frames)
            or any(frame.fold.nunique() < 3 for frame in frames.values())):
        lines += ['**Zwischenstand.** Gepoolte Zeilen können unterschiedliche Teilkohorten umfassen.', '']
    if 'search_progress.csv' in tables:
        progress = tables['search_progress.csv']
        lines.extend([
            f"{progress.inner_fits_checkpointed.sum()}/{progress.inner_fits_expected.sum()} innere Fits gespeichert. "
            f"{progress.outer_reported.sum()}/{len(progress)} äußere Folds ausgewertet. "
            f'{progress.persisted_failure_files.sum()} gespeicherte Fehlerdateien.',
            '[Suchfortschritt je äußerem Fold](search_progress.csv)', '',
        ])
    if not frames:
        lines.append('Es liegen noch keine äußeren OOF-Checkpoints vor.')
        (output / 'status.md').write_text('\n'.join(lines) + '\n')
        return
    rows = tables['decision_metrics.csv']
    lines.extend([
        '## Ergebnisse auf den äußeren Folds',
        '',
        '| Features | Äußere Folds | Entscheidung | Kritisch n / % | 1 → 3 n / % | 2 → 3 n / % | '
        'Fehlalarme % | Freigabe % | Macro-F1 | Balanced Accuracy |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    pooled = rows.loc[rows.scope.eq('pooled')]
    for features in ordered_features(frames):
        folds = ', '.join(map(str, sorted(frames[features].fold.unique())))
        for row in pooled.loc[pooled.features.eq(features) & pooled.method.ne(METHODS['prediction_review'])].itertuples():
            lines.append(
                f'| {FEATURE_NAMES.get(features, features)} | {folds} | {row.method} | '
                f'{row.critical_errors} / {row.critical_error_rate:.2f} | '
                f'{row.critical_1_errors} / {row.critical_1_rate:.2f} | '
                f'{row.critical_2_errors} / {row.critical_2_rate:.2f} | '
                f'{row.false_alarm_rate:.2f} | {row.release_rate:.2f} | {row.macro_f1:.4f} | '
                f'{row.balanced_accuracy:.4f} |'
            )
    reviews = pooled.loc[pooled.method.eq(METHODS['prediction_review'])]
    if not reviews.empty:
        lines += [
            '', '## Explizite Prüfung', '',
            '| Features | Kritisch n / % | 1 → 3 n / % | 2 → 3 n / % | Fehlalarme % | '
            'Freigabe % | Prüfung % | Prüfung wahre 3 % | '
            'Wahre 3 ohne Freigabe % |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ]
        for row in reviews.itertuples():
            lines.append(
                f'| {FEATURE_NAMES.get(row.features, row.features)} | '
                f'{row.critical_errors} / {row.critical_error_rate:.2f} | '
                f'{row.critical_1_errors} / {row.critical_1_rate:.2f} | '
                f'{row.critical_2_errors} / {row.critical_2_rate:.2f} | {row.false_alarm_rate:.2f} | '
                f'{row.release_rate:.2f} | {row.review_rate:.2f} | {row.true3_review_rate:.2f} | '
                f'{row.true3_nonrelease_rate:.2f} |'
            )
    robustness = tables['body_robustness.csv']
    lines.extend(['', '## Robustheit nach Folds und Körpern', ''])
    for features in ordered_features(frames):
        for arm in ('prediction_regular', 'prediction_adapted'):
            method = METHODS[arm]
            fold_rows = rows.loc[rows.features.eq(features) & rows.method.eq(method) & rows.scope.eq('fold')]
            group = robustness.loc[robustness.features.eq(features) & robustness.method.eq(method)].iloc[0]
            rates = ', '.join(f'{row.group} mit {row.critical_error_rate:.2f} %' for row in fold_rows.itertuples())
            lines.append(
                f'- {FEATURE_NAMES.get(features, features)} · {method}. Folds {rates}. '
                f'{group.bodies_critical_under_2pct}/{group.bodies_with_critical_support} Körper mit '
                f'kritischen Messungen liegen unter 2 %. Schlechtester Körper {group.worst_body} '
                f'mit {group.worst_body_critical_errors}/{group.worst_body_critical_support} kritischen Fehlern.'
            )
    if not selections.empty:
        lines.extend(['', '## Auswahl je äußeren Fold', ''])
        columns = ['features', 'fold', 'kind', 'candidate', 'C', 'gamma', 'weights', 'gate',
                   'threshold', 'calibration', 'fallback', 'inner_selection_basis', 'inner_feasible',
                   'inner_critical_error_rate', 'inner_critical_1_rate', 'inner_false_alarm_rate']
        columns = [name for name in columns if name in selections]
        lines.append('| ' + ' | '.join(columns) + ' |')
        lines.append('| ' + ' | '.join(['---'] * len(columns)) + ' |')
        for values in selections[columns].fillna('—').itertuples(index=False, name=None):
            rendered = [f'{value:.6g}' if isinstance(value, float) else str(value) for value in values]
            lines.append('| ' + ' | '.join(rendered) + ' |')
    lines.extend([
        '', '## Artefakte', '',
        '- [Entscheidungsmetriken je Scope](decision_metrics.csv) und [gepaarte Deltas](paired_deltas.csv).',
        '- [Gepaarte Prüfkörper-Resampling-Perzentile](paired_specimen_intervals.csv). Die Spanne beschreibt',
        '  Unterschiede bei festgehaltenen Fits und denselben äußeren Folds.',
        '- [Konfusionsmatrizen](confusion_matrices.csv), [Entscheidungen nach wahrer Klasse](outcomes_by_true_class.csv)',
        '  und [Wahrscheinlichkeitsgüte](probability_metrics.csv).',
        '- [Betriebspunkte](operating_points.png) und [Körperunterschiede](specimen_deltas.png).', '',
    ])
    (output / 'status.md').write_text('\n'.join(lines))


def make_plots(output, frames, tables):
    import matplotlib

    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    metrics = tables['decision_metrics.csv']
    pooled = metrics.loc[metrics.scope.eq('pooled')]
    names = ordered_features(frames)
    methods = [name for col, name in METHODS.items() if col != 'prediction_review']
    palettes = dict(zip(methods, COLORS, strict=False))
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    x = np.arange(len(names))
    for ax, metric, title in zip(axes.flat, ('critical_1_rate', 'critical_2_rate', 'false_alarm_rate', 'macro_f1'),
                                 ('Kritische Fehler 1 → 3', 'Kritische Fehler 2 → 3', 'Fehlalarme 3 → 1/2', 'Macro-F1'),
                                 strict=True):
        for offset, method in enumerate(methods):
            values = pooled.loc[pooled.method.eq(method)].set_index('features')[metric].reindex(names).to_numpy()
            ax.bar(x + (offset - 1.5) * .18, values, .16, color=palettes[method], label=method)
        ax.set_xticks(x, [FEATURE_NAMES.get(name, name) for name in names], fontsize=8)
        ax.set(title=title, ylabel='%' if metric != 'macro_f1' else 'F1')
        ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(output / 'operating_points.png', dpi=140)
    plt.close(fig)

    deltas = tables['paired_deltas.csv']
    bodies = deltas.loc[(deltas.scope.eq('specimen_id')) & deltas.reference.eq('Regulär')
                        & deltas.variant.eq('Adaptiert') & deltas.critical_support.gt(0)]
    specimen_order = sorted(bodies.group.unique())
    fig, ax = plt.subplots(figsize=(13, 5.1), layout='constrained')
    x = np.arange(len(specimen_order))
    for i, name in enumerate(names):
        values = bodies.loc[bodies.features.eq(name)].set_index('group').delta_critical_error_rate
        ax.plot(x, values.reindex(specimen_order), 'o-', label=FEATURE_NAMES.get(name, name), ms=3)
    ax.axhline(0, color='.4', lw=.7)
    ax.set_xticks(x, [name.removeprefix('messreihe_').replace('_pk_', '/') for name in specimen_order],
                  rotation=45, ha='right', fontsize=8)
    ax.set(title='Kritische Fehler je Prüfkörper · adaptiert gegenüber regulär',
           ylabel='Differenz in Prozentpunkten', xlabel='Messreihe / Prüfkörper')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis='y', alpha=.2)
    fig.savefig(output / 'specimen_deltas.png', dpi=140)
    plt.close(fig)


def write_report(output, source, *, finalized=False):
    output, source = Path(output), Path(source)
    cohort = frozen_train(source)[META]
    frames = verified_frames(output, source, cohort)
    selections_path = output / 'selections.csv'
    selections = pd.read_csv(selections_path) if selections_path.is_file() else pd.DataFrame()
    report = output / 'report'
    report.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=report) as scratch:
        scratch = Path(scratch)
        tables = summarize(frames) if frames else {}
        if frames:
            tables['body_robustness.csv'] = body_robustness(tables['decision_metrics.csv'])
            tables['paired_specimen_intervals.csv'] = paired_specimen_intervals(tables['paired_deltas.csv'])
        progress = search_progress(output, frames)
        if not progress.empty:
            tables['search_progress.csv'] = progress
        for filename, table in tables.items():
            table.to_csv(scratch / filename, index=False)
        if frames:
            make_plots(scratch, frames, tables)
        render(scratch, frames, selections, tables, finalized)
        for path in scratch.iterdir():
            path.replace(report / path.name)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=EXPERIMENTS_ROOT / 'specimen_feature_grid_v1')
    parser.add_argument('--finalized', action='store_true')
    args = parser.parse_args()
    write_report(args.output_dir, args.source, finalized=args.finalized)


if __name__ == '__main__':
    main()
