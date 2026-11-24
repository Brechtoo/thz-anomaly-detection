"""Select spatial rules on inner training groups and evaluate on frozen outer folds."""

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from platform import python_version

import mlflow
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.analysis.spatial_postprocessing_report import spatial_metrics, summarize
from src.data.split_manifest import sha256_file
from src.evaluation.spatial_postprocessing import (
    apply_strategy,
    build_neighborhoods,
    candidate_grid,
    summarize_neighbors,
)
from src.experiments.provenance import capture_git_state
from src.experiments.spatial_inputs import load_spatial_input
from src.paths import EXPERIMENTS_ROOT, PROJECT_ROOT

SOURCE = EXPERIMENTS_ROOT / 'specimen_svm_nested_hpo_v1'
OUTPUT = EXPERIMENTS_ROOT / 'specimen_spatial_postprocessing_v1_verified'
BASELINES = ('manual', 'manual+tsfel', 'manual+tsfel+physical_v2')
FAMILIES = ('pointwise', 'smooth', 'rescue', 'consensus')
PROBABILITIES = [f'p_{label}' for label in range(4)]
PREDICTION_KEYS = ['instance_id', 'specimen_id', 'source_dataset', 'fold', 'label', 'x', 'y']

# Changes are absolute fractions relative to the same inner baseline.
MAX_LOSS = {
    'macro_f1': .03, 'balanced_accuracy': .03, 'recall_1': .03,
    'recall_3': .12, 'precision_3': .02, 'release_rate': .10,
}
MAX_INCREASE = {'false_alarm_rate': .12, 'poor_measurement_release_rate': .02}
MAX_INNER_FOLD_RECALL_3_LOSS = .20


def prediction_inputs(frame):
    """Keep labels and fold identifiers out of the transformation."""
    baseline = frame.prediction.to_numpy()
    probabilities = frame[PROBABILITIES].to_numpy()
    points = frame[['specimen_id', 'source_dataset', 'x', 'y']]
    return baseline, probabilities, points


def neighbor_summaries(points, probabilities, candidates):
    keys = {(c.radius_factor, c.weighting) for c in candidates if c.radius_factor is not None}
    geometry = build_neighborhoods(points, sorted({radius for radius, _ in keys}))
    return {
        (radius, weight): summarize_neighbors(geometry[radius], probabilities, weight)
        for radius, weight in sorted(keys)
    }


def admissible(metrics, baseline, fold_recall_3, baseline_fold_recall_3):
    """Require useful safety gains and set limits on the cost of fewer releases."""
    eps = 1e-12
    return (
        metrics['cavity_misses'] <= baseline['cavity_misses']
        and metrics['critical_errors'] <= baseline['critical_errors']
        and (metrics['cavity_misses'], metrics['critical_errors'])
        != (baseline['cavity_misses'], baseline['critical_errors'])
        and all(metrics[key] >= baseline[key] - limit - eps for key, limit in MAX_LOSS.items())
        and all(metrics[key] <= baseline[key] + limit + eps for key, limit in MAX_INCREASE.items())
        and np.all(fold_recall_3 >= baseline_fold_recall_3 - MAX_INNER_FOLD_RECALL_3_LOSS - eps)
    )


def ranking(metrics):
    return (
        metrics['cavity_misses'], metrics['critical_errors'], -metrics['macro_f1'],
        -metrics['balanced_accuracy'], metrics['false_alarm_rate'], -metrics['release_rate'],
    )


def select_rules(frame, candidates):
    """Tune separately within each family, with the native prediction as fallback."""
    baseline, probabilities, points = prediction_inputs(frame)
    summaries = neighbor_summaries(points, probabilities, candidates)
    labels = frame.label.to_numpy()
    baseline_metrics = spatial_metrics(labels, baseline)
    # Recall for class 3 in each pre-existing inner fold guards against concentrating release loss.
    clear = [frame.fold.eq(fold).to_numpy() & (labels == 3) for fold in sorted(frame.fold.unique())]
    if any(not mask.any() for mask in clear):
        raise ValueError('An inner fold has no true class-3 points')
    baseline_recall = np.array([np.mean(baseline[mask] == 3) for mask in clear])
    best = dict.fromkeys(FAMILIES)
    best_key = {}
    trials = []
    for index, candidate in enumerate(candidates):
        summary = summaries.get((candidate.radius_factor, candidate.weighting))
        predicted = apply_strategy(baseline, probabilities, summary, candidate)
        metrics = spatial_metrics(labels, predicted)
        fold_recall = np.array([np.mean(predicted[mask] == 3) for mask in clear])
        allowed = admissible(metrics, baseline_metrics, fold_recall, baseline_recall)
        key = (*ranking(metrics), index)
        family = candidate.strategy
        trials.append({**asdict(candidate), **metrics, 'candidate_index': index, 'admissible': allowed,
                       'worst_fold_recall_3_change': float(np.min(fold_recall - baseline_recall))})
        if allowed and (family not in best_key or key < best_key[family]):
            best[family] = index
            best_key[family] = key
    trials.insert(0, {'strategy': 'baseline', **baseline_metrics, 'candidate_index': -1,
                      'admissible': True, 'worst_fold_recall_3_change': 0.})
    # An optional operational selector can choose a spatial family or retain the baseline.
    best['selected_spatial'] = min(
        (best[family] for family in FAMILIES[1:] if best[family] is not None),
        key=lambda index: (*ranking(trials[index + 1]), index),
        default=None,
    )
    return best, trials


def predict_outer(frame, feature_set, selected, candidates, trials):
    baseline, probabilities, points = prediction_inputs(frame)
    chosen = {name: candidates[index] for name, index in selected.items() if index is not None}
    summaries = neighbor_summaries(points, probabilities, chosen.values())
    predicted = {'baseline': baseline}
    rules = []
    for name, index in selected.items():
        candidate = chosen.get(name)
        if candidate is None:
            predicted[name] = baseline
        else:
            summary = summaries.get((candidate.radius_factor, candidate.weighting))
            predicted[name] = apply_strategy(baseline, probabilities, summary, candidate)
        details = trials[0 if index is None else index + 1]
        rules.append({'baseline': feature_set, 'fold': int(frame.fold.iloc[0]), 'family': name,
                      'fallback': index is None, **details})
    keys = frame[PREDICTION_KEYS].assign(baseline=feature_set, baseline_prediction=baseline)
    return pd.concat([keys.assign(strategy=name, prediction=prediction)
                      for name, prediction in predicted.items()], ignore_index=True), rules


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')


def track(output, report, candidate_count):
    """Use a separate local tracking store so the historical MLflow database stays untouched."""
    report_dir = output / 'report'
    client = mlflow.MlflowClient(tracking_uri=f'sqlite:///{output / "mlflow.db"}')
    experiment = client.create_experiment(
        'thz-specimen-spatial-postprocessing', artifact_location=(output / 'mlartifacts').as_uri(),
    )
    run = client.create_run(experiment, tags={
        'mlflow.runName': output.name, 'scope': 'nested_training_oof', 'holdouts': 'closed',
    })
    run_id = run.info.run_id
    try:
        for key, value in {'candidates': candidate_count, 'outer_folds': 3, 'baselines': len(BASELINES)}.items():
            client.log_param(run_id, key, value)
        for row in report['pooled'].itertuples(index=False):
            baseline_key = row.baseline.replace('+', '__')
            for key in ('cavity_miss_rate', 'critical_error_rate', 'false_alarm_rate',
                        'release_rate', 'macro_f1', 'balanced_accuracy'):
                client.log_metric(run_id, f'{baseline_key}.{row.strategy}.{key}', getattr(row, key))
        for name in ('plan.json', 'selections.csv', 'inner_trials.csv', 'predictions.parquet'):
            client.log_artifact(run_id, str(output / name))
        client.log_artifacts(run_id, str(report_dir), 'report')
    except Exception:
        client.set_terminated(run_id, 'FAILED')
        raise
    client.set_terminated(run_id)
    write_json(output / 'run.json', {'run_id': run_id, 'experiment_id': experiment, 'tracking': 'mlflow.db'})


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source):
        raise ValueError('Write to a new directory outside the historical experiment')
    output.mkdir(parents=True, exist_ok=False)
    candidates = tuple(candidate_grid())
    frames, selections, trials, input_hashes = [], [], [], {}
    with threadpool_limits(limits=1):
        for feature_set in BASELINES:
            folds, hashes = load_spatial_input(source, feature_set)
            input_hashes.update(hashes)
            for fold in folds:
                selected, results = select_rules(fold.inner, candidates)
                predictions, rules = predict_outer(fold.outer, feature_set, selected, candidates, results)
                frames.append(predictions)
                selections.extend({
                    **rule, 'svm_candidate': fold.candidate, 'svm_calibration': fold.calibration,
                } for rule in rules)
                trials.extend({'baseline': feature_set, 'outer_fold': fold.fold, **row} for row in results)
                print(f'{feature_set}: outer fold {fold.fold} evaluated', flush=True)
        predictions = pd.concat(frames, ignore_index=True)
        report = summarize(predictions)

    sources = ['src/experiments/spatial_inputs.py', 'src/evaluation/spatial_postprocessing.py',
               'src/analysis/spatial_postprocessing_report.py', 'src/experiments/spatial_postprocessing.py',
               'src/evaluation/svm_nested_selection.py']
    write_json(output / 'plan.json', {
        'protocol': 'spatial_nested_train_oof_v1',
        'baselines': BASELINES, 'candidates': [asdict(c) for c in candidates],
        'max_inner_loss': MAX_LOSS, 'max_inner_increase': MAX_INCREASE,
        'max_inner_fold_recall_3_loss': MAX_INNER_FOLD_RECALL_3_LOSS,
        'ranking': ['cavity_misses', 'critical_errors', '-macro_f1', '-balanced_accuracy',
                    'false_alarm_rate', '-release_rate', 'grid_order'],
        'source': str(source), 'inputs_sha256': input_hashes,
        'code_sha256': {name: sha256_file(PROJECT_ROOT / name) for name in sources},
        'git': capture_git_state(PROJECT_ROOT),
        'python': python_version(),
        'packages': {name: version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn',
                                                       'pyarrow', 'mlflow')},
    })
    pd.DataFrame(trials).to_csv(output / 'inner_trials.csv', index=False)
    pd.DataFrame(selections).to_csv(output / 'selections.csv', index=False)
    predictions.to_parquet(output / 'predictions.parquet', index=False, compression='zstd')
    report_dir = output / 'report'
    report_dir.mkdir()
    for name, table in report.items():
        table.to_csv(report_dir / f'{name}.csv', index=False)
    track(output, report, len(candidates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.source, args.output_dir)


if __name__ == '__main__':
    main()
