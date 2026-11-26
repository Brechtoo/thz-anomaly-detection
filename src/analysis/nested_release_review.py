"""Training-only review gates and consensus on the frozen nested HPO candidates."""

import argparse
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.analysis.rbf_svm_oof_thresholds import (
    decision_metrics,
    frozen_train,
    scopes,
    threshold_below_target,
)
from src.data.split_manifest import sha256_file
from src.evaluation.svm_nested_selection import (
    body_cross_fitted_probabilities,
    fit_calibrator,
    predict_probabilities,
)
from src.experiments.svm_nested_hpo import (
    FEATURES,
    KEYS,
    SCORES,
    SOURCE,
    pooled_inner,
    read_fit,
)
from src.experiments.tabular_nested_hpo import MODELS, OUTPUT, SVM, verified_assignments
from src.features.specimen_inputs import read_json, require_hash
from src.paths import ANALYSIS_RESULTS

FAMILIES = ('rbf_svm', *MODELS)
ENSEMBLES = {'svm_mlp': ('rbf_svm', 'mlp'), 'all_six': FAMILIES}
STRATEGIES = ('pooled', 'each_fold')


def thresholds(inner, floor):
    y, base, p3 = (inner[k].to_numpy() for k in ('label', 'base', 'p3'))
    pooled = threshold_below_target(y, base, p3)
    folds = [threshold_below_target(y[i], base[i], p3[i]) for i in inner.groupby('fold').indices.values()]
    return {'original_review': floor, 'pooled': max(floor, pooled), 'each_fold': max(floor, *folds)}


def review(base, release):
    return np.where((base == 3) & ~release, -1, base)


def budget_threshold(base, scores, releases):
    """Release at most the requested count on inner data, treating ties as a block."""
    eligible = scores[base == 3]
    if releases >= len(eligible):
        return 0.
    boundary = np.partition(eligible, len(eligible) - releases - 1)[len(eligible) - releases - 1]
    return float(np.nextafter(boundary, np.inf))


def add_scores(frame, hard, p, gate):
    frame = frame[KEYS].copy()
    frame['base'] = hard if gate == 'predict' else p.argmax(axis=1)
    frame['p3'] = p[:, 3]
    return frame


def load_scores(model, combo, outer, train, assignments):
    svm = model == 'rbf_svm'
    work = (SVM / 'search' if svm else OUTPUT / 'fits' / model) / combo / f'outer_{outer}'
    choice = read_json(work / 'selection.json')['adapted']
    candidate, method = choice['candidate'], choice['calibration']
    inner, _ = pooled_inner(work, assignments, train, candidate)
    inner = inner.astype(dict.fromkeys(SCORES, 'float64'))
    p_inner = body_cross_fitted_probabilities(inner, method)
    held = train.loc[train.fold.eq(outer)]
    if svm:
        checkpoint = SVM / 'outer' / combo / f'fold_{outer}'
        path = checkpoint / 'predictions.parquet'
        require_hash(path, read_json(checkpoint / 'provenance.json')['sha256'])
        archived = pd.read_parquet(path)
        p = archived[[f'p_{i}' for i in range(4)]].to_numpy()
    else:
        path = work / 'outer_predictions.parquet'
        require_hash(path, read_json(work / 'outer_summary.json')['sha256'])
        archived = pd.read_parquet(path)
        fit, _ = read_fit(work / 'outer' / candidate, held)
        np.testing.assert_array_equal(fit.prediction, archived.prediction_hard)
        calibrator = fit_calibrator(inner[SCORES].to_numpy(), inner.label.to_numpy(), method)
        p = predict_probabilities(calibrator, fit[SCORES].to_numpy(dtype=float))
    pd.testing.assert_frame_equal(archived[KEYS], held[KEYS].reset_index(drop=True))
    test_fold = add_scores(archived, archived.prediction_hard, p, choice['gate'])
    np.testing.assert_array_equal((test_fold.base == 3) & (test_fold.p3 >= choice['threshold']),
                                  archived.prediction_adapted == 3)
    return add_scores(inner, inner.prediction, p_inner, choice['gate']), test_fold, choice


def assess_fold(combo, outer, train, assignments, selections):
    masks = {}
    result = train.loc[train.fold.eq(outer), KEYS].reset_index(drop=True).copy()
    svm_inner = None
    for model in FAMILIES:
        inner, held, choice = load_scores(model, combo, outer, train, assignments)
        if svm_inner is None:
            svm_inner, svm_held = inner, held
        else:
            pd.testing.assert_frame_equal(inner[KEYS], svm_inner[KEYS])
        for strategy, threshold in thresholds(inner, choice['threshold']).items():
            inner_release = (inner.base == 3) & (inner.p3 >= threshold)
            held_release = (held.base == 3) & (held.p3 >= threshold)
            masks[model, strategy] = inner_release, held_release
            result[f'{model}:{strategy}'] = review(held.base, held_release)
            selections.append(dict(features=combo, outer=outer, model=model, strategy=strategy,
                                   candidate=choice['candidate'], calibration=choice['calibration'],
                                   gate=choice['gate'], threshold=threshold, inner_releases=inner_release.sum()))
    for ensemble, members in ENSEMBLES.items():
        for strategy in STRATEGIES:
            inside, outside = (np.logical_and.reduce(parts) for parts in zip(*(masks[m, strategy] for m in members)))
            result[f'{ensemble}:{strategy}'] = review(svm_held.base, outside)
            threshold = budget_threshold(svm_inner.base.to_numpy(), svm_inner.p3.to_numpy(), inside.sum())
            comparator = f'svm_budget_{ensemble}'
            result[f'{comparator}:{strategy}'] = review(svm_held.base, svm_held.p3 >= threshold)
            selections.append(dict(features=combo, outer=outer, model=comparator, strategy=strategy,
                                   threshold=threshold, inner_releases=((svm_inner.base == 3)
                                                                        & (svm_inner.p3 >= threshold)).sum(),
                                   ensemble_inner_releases=inside.sum()))
    return result


def summarize(frame, combo, metrics, overlaps):
    for scope, group, idx in scopes(frame):
        part = frame.iloc[idx]
        for col in frame.columns.difference(KEYS):
            model, strategy = col.split(':')
            pred = part[col].to_numpy()
            # This analysis has a review outcome. Use the historical denominator definitions.
            values = decision_metrics(part.label.to_numpy(), pred, pred)
            fields = ('n', 'critical_support', 'critical_1_errors', 'critical_2_errors', 'critical_errors',
                      'critical_error_rate', 'false_alarm_rate', 'review_rate', 'review', 'clear_rate',
                      'true3_review_rate', 'true3_nonclear_rate')
            metrics.append(dict(features=combo, scope=scope, group=group, model=model, strategy=strategy,
                                **{key: values[key] for key in fields}))
    for strategy in ('original_review', *STRATEGIES):
        errors = {m: frame.label.isin([1, 2]) & frame[f'{m}:{strategy}'].eq(3) for m in FAMILIES}
        for a, b in combinations(FAMILIES, 2):
            shared, union = (errors[a] & errors[b]).sum(), (errors[a] | errors[b]).sum()
            overlaps.append(dict(features=combo, strategy=strategy, a=a, b=b, shared=shared,
                                 only_a=errors[a].sum()-shared, only_b=errors[b].sum()-shared,
                                 jaccard=shared / union if union else np.nan))


def run(output):
    if output.exists():
        raise FileExistsError(output)
    train = frozen_train(SOURCE)
    assignments = verified_assignments(train, SVM / 'inner_assignments.parquet',
                                       read_json(SVM / 'plan.json')['inner_assignment_sha256'])
    metrics, overlaps, selections = [], [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp, threadpool_limits(limits=1):
        stage = Path(temp) / 'report'
        stage.mkdir()
        for combo in FEATURES:
            folds = [assess_fold(combo, k, train, assignments.loc[assignments.outer.eq(k)], selections)
                     for k in range(3)]
            frame = pd.concat(folds, ignore_index=True).sort_values('instance_id', ignore_index=True)
            pd.testing.assert_frame_equal(frame[KEYS], train[KEYS])
            summarize(frame, combo, metrics, overlaps)
            print(f'{combo}: verified and assessed', flush=True)
        summary = pd.DataFrame(metrics)
        summary.to_csv(stage / 'metrics.csv', index=False)
        pd.DataFrame(overlaps).to_csv(stage / 'overlap.csv', index=False)
        pd.DataFrame(selections).to_csv(stage / 'thresholds.csv', index=False)
        stage.rename(output)
    mlflow.set_tracking_uri(f'sqlite:///{output.resolve() / "mlflow.db"}')
    client = mlflow.MlflowClient()
    experiment = client.create_experiment('nested-release-review', artifact_location=(output.resolve() / 'artifacts').as_uri())
    with mlflow.start_run(experiment_id=experiment):
        mlflow.log_params({'svm_plan': sha256_file(SVM / 'plan.json'), 'tabular_plan': sha256_file(OUTPUT / 'plan.json'),
                           'source_sha256': sha256_file(Path(__file__)), 'seed': 42, 'target': 'strictly below 2%',
                           'ensembles': str(ENSEMBLES), 'fallback': 'review'})
        for row in summary.loc[summary.scope.eq('pooled')].itertuples():
            prefix = f'{row.features.replace("+","_")}.{row.model}.{row.strategy}'
            mlflow.log_metrics({f'{prefix}.{name}': getattr(row, name)
                                for name in ('critical_error_rate', 'false_alarm_rate', 'review_rate', 'clear_rate')})
        for path in output.glob('*.csv'):
            mlflow.log_artifact(str(path))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, default=ANALYSIS_RESULTS / 'nested_release_review_v1')
    run(cli.parse_args().output)
