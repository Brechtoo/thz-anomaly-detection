"""Tune an RBF SVM on frozen training-body folds, then evaluate the frozen holdouts."""
import argparse
import importlib.metadata
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits

from src.data.split_manifest import SPLIT_NAMES, restore_splits, sha256_file
from src.evaluation.metrics import ERROR_METRICS_VERSION
from src.evaluation.reporting import evaluate_holdout
from src.experiments.anchor_group_cv import fold_pipeline, scores
from src.experiments.model_registry import create_model
from src.experiments.provenance import capture_git_state
from src.experiments.tracking import set_project_experiment
from src.experiments.training import fit_and_predict, shared_rbf_candidates
from src.features.extract_tsfresh_features import select_training_features
from src.features.specimen_inputs import (
    ANCHOR,
    EARLY,
    FOLDS,
    SPLITS,
    TSFRESH,
    early_holdouts,
    load_cohorts,
    load_early_manual_train,
    load_tsfresh_holdout,
    raw_tsfresh,
    tsfresh_manifest,
    write_json,
)
from src.paths import PROJECT_ROOT

SEED = 42
GRID = list(ParameterGrid({
    'model__C': [0.3, 3.0],
    'model__gamma': [0.001, 'scale'],
    'model__class_weight': [None, 'balanced'],
}))


def select_candidate(truth, oof):
    """Rank pooled counts first, then macro-F1 and false alarms on the same points."""
    results = pd.DataFrame([
        {'candidate': i, **scores(truth, prediction)} for i, prediction in enumerate(oof)
    ])
    return results.sort_values(
        ['critical_errors', 'macro_f1', 'false_alarms', 'candidate'],
        ascending=[True, False, True, True],
    )


def tune(features, train, kind, output, jobs):
    """Make one out-of-fold prediction per training point and candidate."""
    labels = train.label
    oof = np.full((len(GRID), len(train)), -1)
    rows = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for fold in sorted(train.fold.unique()):
            val = train.fold.eq(fold).to_numpy()
            start = time.perf_counter()
            # Learn tsfresh imputation and label-based feature selection on the
            # fit specimens, then use the same columns for every candidate.
            matrix = (select_training_features(features, labels, train.index[~val])
                      if kind == 'tsfresh' else features)
            prep_seconds = time.perf_counter() - start
            if kind == 'tsfresh':
                write_json(output / f'fold_{fold}_features.json', matrix.columns.tolist())
            X = np.ascontiguousarray(matrix, dtype=np.float64)
            X_train, X_val, y_train = X[~val], X[val], labels[~val].to_numpy()
            print(f'{kind} fold {fold}: {X.shape[1]} input features, {prep_seconds:.1f}s prep', flush=True)

            def run_candidate(item):
                candidate, params = item
                model = svm(kind, params)
                predictions, fit, durations = fit_and_predict(model, X_train, y_train, {'val': X_val})
                prediction = predictions['val']
                print(f'{kind} fold {fold} candidate {candidate}: {fit:.1f}s fit', flush=True)
                return candidate, prediction, dict(
                    training_seconds=fit, prediction_seconds=durations['val'],
                    support_vectors=int(model[-1].n_support_.sum()),
                )

            if kind == 'tsfresh':
                candidates = shared_rbf_candidates(X_train, X_val, y_train, pool, output, grid=GRID, seed=SEED)
            else:
                candidates = pool.map(run_candidate, enumerate(GRID))
            for i, prediction, timings in candidates:
                oof[i, val] = prediction
                rows.append(dict(fold=int(fold), candidate=i, shared_preprocessing_seconds=prep_seconds,
                                 input_features=X.shape[1], **timings, **scores(labels[val], prediction)))
            pd.DataFrame(rows).to_csv(output / 'fold_metrics.csv', index=False)
    if not np.isin(oof, [0, 1, 2, 3]).all():
        raise ValueError('Incomplete OOF predictions')
    ranking = select_candidate(labels, oof)
    ranking.to_csv(output / 'cv_ranking.csv', index=False)
    selected = int(ranking.iloc[0].candidate)
    write_json(output / 'selected.json', {'candidate': selected, 'parameters': GRID[selected]})
    evaluate_holdout(train, oof[selected], output / 'oof')
    return selected, ranking


def svm(kind, params):
    model = create_model('rbf_svm', seed=SEED)
    model.set_params(model__cache_size=768, **params)
    return fold_pipeline(model) if kind == 'manual52_early' else model


def evaluate_final(kind, params, matrix, membership, output):
    frames = restore_splits(matrix, membership.label, membership.instance_id, SPLITS)
    X_train, X_val, X_test = [frame.to_numpy(dtype=np.float64) for frame in frames[:3]]
    evaluations = {'validation': X_val, 'test': X_test}
    # Early-anchor holdout comparators use exactly the same inputs and fold-style
    # imputation, with the existing, fixed registry parameters.
    models = {'rbf_svm': svm(kind, params)}
    if kind == 'manual52_early':
        models.update({name: fold_pipeline(create_model(name, seed=SEED)) for name in ('hgb', 'mlp')})
    rows = []
    write_json(output / 'final_features.json', matrix.columns.tolist())
    for name, model in models.items():
        predictions, fit, durations = fit_and_predict(model, X_train, frames[3], evaluations)
        path = output / f'{name}.joblib'
        joblib.dump(model, path, compress=3)
        reloaded = joblib.load(path)
        print(f'{kind} final {name}: {fit:.1f}s fit', flush=True)
        with mlflow.start_run(run_name=f'{kind}-{name}-holdout') as run:
            mlflow.log_params({k: str(v) for k, v in model.get_params(deep=True).items()})
            mlflow.log_metric('training_seconds', fit)
            if name == 'rbf_svm':
                mlflow.log_metric('support_vectors', int(model[-1].n_support_.sum()))
            mlflow.log_artifact(str(path))
            mlflow.log_artifact(str(output / 'plan.json'))
            mlflow.log_artifact(str(output / 'final_features.json'))
            for split, frame in zip(SPLIT_NAMES[1:], frames[1:3]):
                np.testing.assert_array_equal(reloaded.predict(evaluations[split]), predictions[split])
                meta = membership.loc[frame.index]
                destination = output / 'holdout' / name / split
                metrics = evaluate_holdout(
                    meta, predictions[split], destination, include_critical_breakdown=True,
                )
                metrics['prediction_seconds'] = durations[split]
                mlflow.log_metrics({f'{split}_{k}': v for k, v in metrics.items()})
                mlflow.log_artifacts(str(destination), artifact_path=split)
                rows.append(dict(model=name, split=split, run_id=run.info.run_id, training_seconds=fit,
                                 model_bytes=path.stat().st_size, **metrics))
        pd.DataFrame(rows).to_csv(output / 'holdout_metrics.csv', index=False)


def run(kind, output, jobs):
    if mlflow.active_run() is not None or os.environ.get('MLFLOW_RUN_ID'):
        raise ValueError('Fresh local MLflow runs required')
    started = time.perf_counter()
    membership, train = load_cohorts()
    ts = tsfresh_manifest()
    output.mkdir(parents=True, exist_ok=False)
    inputs = [SPLITS / 'manifest.json', FOLDS / 'manifest.json', Path(__file__)]
    if kind == 'manual52_early':
        path = ANCHOR / 'features' / f'{EARLY}.npz'
        features = load_early_manual_train(train)
        inputs.extend([path, ANCHOR / 'extraction_manifest.json'])
    else:
        features = raw_tsfresh(membership, train, ts)
        inputs.append(TSFRESH / 'manifest.json')
    write_json(output / 'plan.json', {
        'feature_set': kind, 'seed': SEED, 'error_metrics_version': ERROR_METRICS_VERSION,
        'grid': GRID, 'kernel_cache_mb': 768, 'candidate_jobs': jobs, 'threads_per_fit': 1,
        'cv_kernel': 'shared precomputed RBF (192 MiB solver cache)' if kind == 'tsfresh' else 'direct RBF',
        'rank_by': ['pooled critical_errors ascending', 'pooled macro_f1 descending',
                    'pooled false_alarms ascending', 'candidate id ascending'],
        'final_fit': 'training bodies only', 'split_roles': 'validation and test for evaluation only',
        'python': platform.python_version(), 'platform': platform.platform(), 'logical_cpus': os.cpu_count(),
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'inputs_sha256': {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in inputs},
        'git': capture_git_state(PROJECT_ROOT),
    })
    (output / 'svm_group_cv.py').write_bytes(Path(__file__).read_bytes())
    set_project_experiment('svm-specimen')
    with threadpool_limits(1):
        before_cv = time.perf_counter()
        selected, ranking = tune(features, train, kind, output, jobs)
        cv_seconds = time.perf_counter() - before_cv
        with mlflow.start_run(run_name=f'{kind}-group-cv') as cv_run:
            mlflow.log_params({'selected_candidate': selected, **GRID[selected]})
            metrics = ranking.iloc[0].to_dict()
            mlflow.log_metrics({'cv_wall_seconds': cv_seconds, **metrics})
            for name in ['plan.json', 'svm_group_cv.py', 'selected.json', 'fold_metrics.csv', 'cv_ranking.csv']:
                mlflow.log_artifact(str(output / name))
            for path in sorted(output.glob('fold_*_features.json')):
                mlflow.log_artifact(str(path))
            mlflow.log_artifacts(str(output / 'oof'), artifact_path='oof')
        # Selection is fixed before preparing the validation and test features.
        prep_start = time.perf_counter()
        matrix = (early_holdouts(output, membership, features, features.columns.tolist(), jobs)
                  if kind == 'manual52_early' else load_tsfresh_holdout(membership, ts))
        prep_seconds = time.perf_counter() - prep_start
        evaluate_final(kind, GRID[selected], matrix, membership, output)
    write_json(output / 'runtime.json', {
        'cv_run_id': cv_run.info.run_id, 'cv_wall_seconds': cv_seconds,
        'final_feature_preparation_seconds': prep_seconds,
        'total_wall_seconds': time.perf_counter() - started,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feature_set', choices=['manual52_early', 'tsfresh'])
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--jobs', default=2, type=int)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    run(args.feature_set, args.output_dir, args.jobs)


if __name__ == '__main__':
    main()
