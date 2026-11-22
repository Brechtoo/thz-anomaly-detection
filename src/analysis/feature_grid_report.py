"""Summarize completed feature-grid OOF checkpoints on the frozen cohort."""
import argparse
import json
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix, log_loss

from src.evaluation.metrics import critical_breakdown
from src.evaluation.reporting import metrics_for_group
from src.features.specimen_inputs import read_json

META = ('instance_id', 'specimen_id', 'source_dataset', 'label', 'fold')
CLASSES = (0, 1, 2, 3)
PROBABILITIES = tuple(f'p_{label}' for label in CLASSES)
RANK = ['critical_errors', 'false_alarms', 'macro_f1', 'balanced_accuracy', 'combination', 'model']
DELTA_METRICS = (
    'critical_errors', 'critical_error_rate', 'critical_1_errors', 'critical_1_rate',
    'critical_2_errors', 'critical_2_rate', 'false_alarms', 'false_alarm_rate',
    'macro_f1', 'balanced_accuracy', 'training_seconds',
)
FOCUS_CHANGES = ('add_tsfel', 'add_physical_v1', 'add_physical_v2', 'physical_v1_to_v2')
HOLDOUT_METRICS = (*DELTA_METRICS, 'accuracy', 'weighted_f1', 'prediction_seconds', 'run_id')
HOLDOUT_REQUIRED = ('critical_errors', 'critical_error_rate', 'false_alarms', 'false_alarm_rate',
                    'macro_f1', 'balanced_accuracy')


def read_fold(folder, reference):
    prediction_file = folder / 'predictions.parquet'
    metrics_file = folder / 'metrics.json'
    frame = pd.read_parquet(prediction_file).sort_values('instance_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[list(META)], reference[list(META)], check_dtype=False)
    if not frame.prediction.isin(CLASSES).all():
        raise ValueError('Predicted labels outside the four-class protocol')
    probabilities = frame[list(PROBABILITIES)].to_numpy(dtype=float)
    if (not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
            or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError('Invalid OOF class probabilities')
    timings = read_json(metrics_file)
    seconds = [float(timings[key]) for key in ('training_seconds', 'prediction_seconds')]
    if not np.isfinite(seconds).all() or min(seconds) < 0:
        raise ValueError('Invalid fit or prediction time')
    return frame, timings


def pooled_metrics(folds, timings):
    oof = pd.concat(folds, ignore_index=True)
    y, pred = oof.label, oof.prediction
    with warnings.catch_warnings():
        # The legacy helper also produces a discarded classification report.
        warnings.simplefilter('ignore', UndefinedMetricWarning)
        metrics = metrics_for_group(y, pred)
    metrics.update(critical_breakdown(y, pred))
    metrics.update(
        points=len(oof),
        log_loss=log_loss(y, oof[list(PROBABILITIES)].to_numpy(dtype=float), labels=CLASSES),
        training_seconds=sum(float(timing['training_seconds']) for timing in timings),
        prediction_seconds=sum(float(timing['prediction_seconds']) for timing in timings),
        reused_folds=sum(bool(timing.get('reused', False)) for timing in timings),
    )
    matrix = confusion_matrix(y, pred, labels=CLASSES)
    cells = [
        {'true_label': label, 'predicted_label': output, 'count': int(matrix[label, output])}
        for label in CLASSES for output in CLASSES
    ]
    return metrics, cells


def paired_deltas(pooled, combinations):
    """Compare each single feature addition and each V1→V2 substitution."""
    by_features = {frozenset(features): slug for slug, features in combinations.items()}
    pairs = []
    for model, records in pooled.groupby('model', sort=False):
        rows = records.set_index('combination').to_dict('index')
        for target, after in rows.items():
            target_features = frozenset(combinations[target])
            contrasts = [
                (target_features - {feature}, f'add_{feature}') for feature in sorted(target_features)
            ]
            if 'physical_v2' in target_features:
                contrasts.append((target_features - {'physical_v2'} | {'physical_v1'}, 'physical_v1_to_v2'))
            for base_features, change in contrasts:
                base = by_features.get(base_features)
                if base not in rows:
                    continue
                before = rows[base]
                pairs.append({
                    'model': model,
                    'baseline': base, 'variant': target, 'change': change,
                    **{f'delta_{key}': after[key] - before[key] for key in DELTA_METRICS},
                })
    return pd.DataFrame(pairs, columns=[
        'model', 'baseline', 'variant', 'change', *(f'delta_{key}' for key in DELTA_METRICS),
    ])


def affected_pairs(pairs):
    """Compare safety first and false alarms only on critical-error ties."""
    critical, alarms = pairs.delta_critical_errors, pairs.delta_false_alarms
    wins = critical.lt(0) | (critical.eq(0) & alarms.lt(0))
    losses = critical.gt(0) | (critical.eq(0) & alarms.gt(0))
    return pairs.assign(outcome=np.select([wins, losses], ['win', 'loss'], default='tie'))


def read_holdouts(output):
    """Read recorded selections and finished holdout metrics, without ranking by test."""
    paths = {'validation': output / 'shortlist.json', 'test': output / 'test_selection.json'}
    selections = {split: read_json(path) for split, path in paths.items() if path.is_file()}
    recorded = {
        'validation': {tuple(pair) for pair in selections.get('validation', {}).get('pairs', [])},
        'test': {tuple(selections['test']['pair'])} if 'test' in selections else set(),
    }
    rows = []
    for split in paths:
        observed = {
            (path.parents[2].name, path.parents[1].name)
            for path in (output / 'holdout').glob(f'*/*/{split}/metrics.json')
        }
        for combination, model in sorted(recorded[split] | observed):
            is_recorded = (combination, model) in recorded[split]
            metric_path = output / 'holdout' / combination / model / split / 'metrics.json'
            failure = output / 'failures' / combination / model / f'{split}.json'
            row = {
                'split': split, 'combination': combination, 'model': model,
                'selection': selections[split]['source'] if is_recorded else 'unrecorded',
                'status': 'pending', 'detail': '',
            }
            if metric_path.is_file():
                try:
                    metrics = read_json(metric_path)
                    if not all(np.isfinite(metrics[key]) for key in HOLDOUT_REQUIRED):
                        raise ValueError('Invalid saved holdout metrics')
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    row.update(status='invalid', detail=str(exc))
                else:
                    row.update({key: metrics.get(key) for key in HOLDOUT_METRICS}, status='finished')
            elif failure.is_file():
                row.update(status='failed', detail=failure.read_text().strip())
            rows.append(row)
    return pd.DataFrame(rows, columns=[
        'split', 'combination', 'model', 'selection', 'status', 'detail', *HOLDOUT_METRICS,
    ]), selections


def render_summary(pooled, status, pairs, holdouts, selections):
    counts = status.status.value_counts()
    planned_pairs = status[['combination', 'model']].drop_duplicates().shape[0]
    lines = [
        '# Stand des HeRAI-Feature-Vergleichs',
        '',
        'Die Rangfolge nutzt gepoolte OOF-Vorhersagen der eingefrorenen Trainingskörper-Folds.',
        'Maßgeblich sind zuerst kritische Fehler (1/2 → 3) und danach Fehlalarme (3 → 1/2).',
        'Macro-F1 und Balanced Accuracy lösen weitere Gleichstände auf.',
        '',
        f'- {int(counts.get("finished", 0))} von {len(status)} geplanten Folds geprüft.',
        f'- {len(pooled)} von {planned_pairs} Modell-Kombinations-Paaren vollständig.',
        f'- {int(counts.get("failed", 0))} Folds mit Fehlerprotokoll und '
        f'{int(counts.get("invalid", 0))} ungültige Checkpoints.',
        '',
    ]
    if pooled.empty:
        lines.append('Eine Rangfolge ist nach dem ersten vollständigen Paar verfügbar.')
        return '\n'.join(lines) + '\n'

    def table(title, rows):
        lines.extend([
            f'## {title}',
            '',
            '| Rang | Features | Modell | Kritisch (n / %) | 1 → 3 (n / %) | 2 → 3 (n / %) | '
            'Fehlalarme (%) | Macro-F1 | Balanced Accuracy | Fit (s) |',
            '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ])
        for row in rows.itertuples(index=False):
            lines.append(
                f'| {row.rank} | {row.combination} | {row.model} | '
                f'{row.critical_errors} / {row.critical_error_rate:.2f} | '
                f'{row.critical_1_errors} / {row.critical_1_rate:.2f} | '
                f'{row.critical_2_errors} / {row.critical_2_rate:.2f} | '
                f'{row.false_alarm_rate:.2f} | {row.macro_f1:.4f} | '
                f'{row.balanced_accuracy:.4f} | {row.training_seconds:.1f} |'
            )
        lines.append('')

    table('Führende abgeschlossene Konfigurationen', pooled.head(12))
    singles = pooled.loc[pooled.feature_count.eq(1)].drop_duplicates('combination')
    multiples = pooled.loc[pooled.feature_count.gt(1)]
    if not singles.empty and not multiples.empty:
        single, multi = singles.iloc[0], multiples.iloc[0]
        lines.append(
            f'Das beste abgeschlossene Einzelset ist **{single.combination} / {single.model}** '
            f'({single.critical_errors} kritische Fehler, {single.false_alarms} Fehlalarme). '
            f'Die beste abgeschlossene Kombination ist **{multi.combination} / {multi.model}** '
            f'({multi.critical_errors} kritische Fehler, {multi.false_alarms} Fehlalarme).'
        )
        lines.append('')
    if not singles.empty:
        table('Bestes abgeschlossenes Modell pro Einzelset', singles)
    lines.extend([
        '## Gepaarte Änderungen bei identischem Modell',
        '',
        'Ein Gewinn bedeutet weniger kritische Fehler oder bei Gleichstand weniger Fehlalarme. '
        'Mehrere Vergleiche nutzen dieselben Körper und bilden keine unabhängigen Beobachtungen.',
        '',
        '| Änderung | Vergleichspaare | Gewinne | Gleichstände | Verluste |',
        '| --- | ---: | ---: | ---: | ---: |',
    ])
    changes = {
        'add_tsfel': 'TSFEL ergänzen',
        'add_physical_v1': 'Physikalisch V1 ergänzen',
        'add_physical_v2': 'Physikalisch V2 ergänzen',
        'physical_v1_to_v2': 'Physikalisch V1 durch V2 ersetzen',
    }
    for change in FOCUS_CHANGES:
        outcomes = pairs.loc[pairs.change.eq(change), 'outcome'].value_counts()
        lines.append(
            f'| {changes[change]} | {outcomes.sum()} | {outcomes.get("win", 0)} | '
            f'{outcomes.get("tie", 0)} | {outcomes.get("loss", 0)} |'
        )
    lines.append('')
    for direction, title in [('win', 'Stärkste Verbesserungen'), ('loss', 'Stärkste Verschlechterungen')]:
        extreme = pairs.loc[pairs.outcome.eq(direction)].sort_values(
            ['delta_critical_errors', 'delta_false_alarms'], ascending=direction == 'win', kind='stable',
        ).head(3)
        if extreme.empty:
            continue
        lines.extend([f'### {title} im gepaarten Vergleich', ''])
        for pair in extreme.itertuples(index=False):
            lines.append(
                f'- {pair.model}, {pair.baseline} → {pair.variant}, '
                f'{pair.delta_critical_errors:+d} kritische Fehler und {pair.delta_false_alarms:+d} Fehlalarme.'
            )
        lines.append('')
    if selections or not holdouts.empty:
        lines.extend([
            '## Holdout',
            '',
            'Die Validation-Shortlist stammt aus der OOF-Rangfolge. Der Test dokumentiert '
            'den auf Validation ausgewählten Kandidaten deskriptiv.',
            '',
            '| Split | Features | Modell | Status | Kritisch (n / %) | Fehlalarme (%) | Macro-F1 | Balanced Accuracy |',
            '| --- | --- | --- | --- | ---: | ---: | ---: | ---: |',
        ])
        for row in holdouts.itertuples(index=False):
            metrics = '— | — | — | —'
            if row.status == 'finished':
                metrics = (
                    f'{row.critical_errors:.0f} / {row.critical_error_rate:.2f} | '
                    f'{row.false_alarm_rate:.2f} | {row.macro_f1:.4f} | {row.balanced_accuracy:.4f}'
                )
            role = 'Test' if row.split == 'test' else 'Validation'
            lines.append(f'| {role} | {row.combination} | {row.model} | {row.status} | {metrics} |')
        lines.append('')
        if 'validation' in selections:
            lines.append(f'Die OOF-Shortlist enthält {len(selections["validation"]["pairs"])} Kandidaten.')
        if 'test' in selections:
            combination, model = selections['test']['pair']
            lines.append(f'Vor der Testausführung wurde **{combination} / {model}** ausgewählt.')
        lines.append('')
    lines.append(
        'Die vollständige Rangfolge und gepaarten Deltas stehen in '
        '[pooled_oof_metrics.csv](pooled_oof_metrics.csv) und [paired_deltas.csv](paired_deltas.csv).'
    )
    return '\n'.join(lines) + '\n'


def write_report(output, combinations, models, cohort):
    """Publish a live summary while the training worker writes independent fold directories."""
    output = Path(output)
    if (cohort.instance_id.isna().any() or not cohort.instance_id.is_unique
            or cohort[['specimen_id', 'source_dataset', 'label', 'fold']].isna().any().any()
            or not cohort.label.isin(CLASSES).all() or set(cohort.fold) != {0, 1, 2}
            or cohort.groupby('specimen_id').fold.nunique().ne(1).any()):
        raise ValueError('Unexpected frozen OOF cohort')
    expected = {
        fold: cohort.loc[cohort.fold.eq(fold)].sort_values('instance_id').reset_index(drop=True)
        for fold in (0, 1, 2)
    }
    pooled_rows, status_rows, matrix_rows = [], [], []
    for combination, features in combinations.items():
        for model in models:
            frames, timings = [], []
            identity = {'combination': combination, 'model': model}
            for fold, reference in expected.items():
                base = output / 'fits' / combination / model / f'fold_{fold}'
                failure = output / 'failures' / combination / model / f'fold_{fold}.json'
                status = {'status': 'pending', 'detail': '', 'run_id': '', 'reused': False}
                if base.exists():
                    try:
                        frame, timing = read_fold(base, reference)
                    except (AssertionError, KeyError, OSError, ValueError) as exc:
                        status.update(status='invalid', detail=str(exc))
                    else:
                        status.update(status='finished', run_id=timing.get('run_id', ''),
                                      reused=timing.get('reused', False))
                        frames.append(frame)
                        timings.append(timing)
                elif failure.is_file():
                    status.update(status='failed', detail=failure.read_text().strip())
                status_rows.append({**identity, 'fold': fold, **status})
            if len(frames) == len(expected):
                metrics, cells = pooled_metrics(frames, timings)
                pooled_rows.append({**identity, 'feature_count': len(features), **metrics})
                matrix_rows.extend({**identity, **cell} for cell in cells)

    pooled = pd.DataFrame(pooled_rows)
    if pooled.empty:
        pooled = pd.DataFrame(columns=['combination', 'model', 'feature_count', *DELTA_METRICS])
    else:
        pooled = pooled.sort_values(RANK, ascending=[True, True, False, False, True, True])
    pooled.insert(0, 'rank', range(1, len(pooled) + 1))
    status = pd.DataFrame(status_rows)
    pairs = affected_pairs(paired_deltas(pooled, combinations))
    holdouts, selections = read_holdouts(output)
    tables = {
        'pooled_oof_metrics.csv': pooled,
        'fold_status.csv': status,
        'paired_deltas.csv': pairs,
        'holdout_metrics.csv': holdouts,
        'confusion_matrices.csv': pd.DataFrame(
            matrix_rows, columns=['combination', 'model', 'true_label', 'predicted_label', 'count'],
        ),
    }
    report_dir = output / 'report'
    report_dir.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=report_dir) as scratch:
        scratch = Path(scratch)
        for filename, table in tables.items():
            table.to_csv(scratch / filename, index=False)
        (scratch / 'status.md').write_text(render_summary(pooled, status, pairs, holdouts, selections))
        for file in scratch.iterdir():
            file.replace(report_dir / file.name)
    return pooled, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    plan = read_json(output / 'plan.json')
    cohort = pd.read_parquet(output / 'cohort.parquet')
    pooled, status = write_report(output, plan['combinations'], plan['models'], cohort)
    print(json.dumps({'completed_pairs': len(pooled), 'fold_status': status.status.value_counts().to_dict()}))


if __name__ == '__main__':
    main()
