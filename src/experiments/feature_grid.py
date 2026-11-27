"""Checkpointed fixed-model grid on the frozen specimen group folds."""
import argparse
import fcntl
import importlib.metadata
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from src.data.split_manifest import sha256_file
from src.evaluation.metrics import ERROR_METRICS_VERSION, critical_breakdown
from src.evaluation.reporting import metrics_for_group
from src.experiments.model_registry import create_model
from src.experiments.specimen_fusion import SEED, SVM_PARAMS
from src.experiments.tracking import set_project_experiment
from src.experiments.training import fit_and_predict, shared_rbf_candidates
from src.features.specimen_inputs import (
    FOLDS,
    SPLITS,
    load_cohorts,
    read_json,
    require_hash,
    write_json,
)
from src.paths import EXPERIMENTS_ROOT, PROJECT_ROOT

MODELS = ('logistic_regression', 'extra_trees', 'mlp', 'lightgbm', 'hgb',
          'xgboost', 'random_forest', 'rbf_svm')
SETS = ('manual', 'tsfresh', 'tsfel', 'physical_v1', 'physical_v2')
LABELS = np.arange(4)
KEYS = ['instance_id', 'specimen_id', 'source_dataset', 'label', 'fold']
EXPERIMENT = 'thz-specimen-feature-grid'
SHORTLIST = 5


def feature_combinations():
    return {'+'.join(group): list(group) for size in range(1, len(SETS))
            for group in combinations(SETS, size) if not {'physical_v1', 'physical_v2'} <= set(group)}


def probabilities_in_label_order(probabilities, classes):
    values = np.asarray(probabilities)
    classes = np.asarray(classes)
    if not np.array_equal(np.sort(classes), LABELS) or values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f'Invalid probability classes or dimensions: {classes}, {values.shape}')
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError('Probabilities outside [0,1] or nonfinite')
    np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0)
    return values[:, np.argsort(classes)]


def pipeline(name):
    model = create_model(name, seed=SEED)
    if name == 'rbf_svm':
        model.set_params(model__probability=True, model__cache_size=192, **SVM_PARAMS)
    return model


def plan_config():
    paths = [
        SPLITS / 'manifest.json', FOLDS / 'manifest.json', Path(__file__),
        PROJECT_ROOT / 'src/features/specimen_grid.py',
        PROJECT_ROOT / 'src/experiments/model_registry.py',
        PROJECT_ROOT / 'src/experiments/training.py',
        PROJECT_ROOT / 'src/features/extract_tsfresh_features.py',
    ]
    return {
        'combinations': feature_combinations(),
        'models': {name: {key: str(val) for key, val in pipeline(name).get_params().items()} for name in MODELS},
        'model_order': list(MODELS),
        'seed': SEED, 'classes': LABELS.tolist(), 'error_metrics': ERROR_METRICS_VERSION,
        'cv_folds': [0, 1, 2], 'threads_per_fit': 1, 'holdout_shortlist_size': SHORTLIST,
        'rank': ['critical_errors ascending', 'false_alarms ascending', 'feature_set/model lexical tie-break'],
        'tsfresh': 'efficient raw 777, original selection and imputation refit per outer training fold',
        'svm_probability': 'native LibSVM probability estimates using outer-train-only internal calibration; hard predict unchanged',
        'source_sha256': {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths},
        'packages': {package: importlib.metadata.version(package) for package in
                     ('numpy', 'pandas', 'scikit-learn', 'scipy', 'tsfresh', 'mlflow', 'xgboost', 'lightgbm', 'pyarrow')},
    }


def fit_arrays(name, training, evaluation, scratch):
    X, y = training
    model = pipeline(name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        start = time.perf_counter()
        if name == 'rbf_svm':
            with ThreadPoolExecutor(max_workers=1) as pool:
                (_, prediction, result), = shared_rbf_candidates(
                    X, evaluation, y, pool, scratch, grid=[SVM_PARAMS], seed=SEED, probabilities=True,
                )
            probabilities = result.pop('probabilities')
            classes = result.pop('probability_classes')
            result['training_seconds'] += result['train_kernel_seconds']
            result['prediction_seconds'] += result['validation_kernel_seconds']
        else:
            preds, fit, pred_time = fit_and_predict(model, X, y, {'evaluation': evaluation})
            prediction = preds['evaluation']
            infer = time.perf_counter()
            probabilities, classes = model.predict_proba(evaluation), model.classes_
            result = {'training_seconds': fit, 'prediction_seconds': pred_time['evaluation'] + time.perf_counter()-infer}
        result['wall_seconds'] = time.perf_counter()-start
    return prediction, probabilities_in_label_order(probabilities, classes), result, sorted({str(w.message) for w in caught})


def metrics_for(frame):
    result = metrics_for_group(frame.label, frame.prediction)
    result.update(critical_breakdown(frame.label, frame.prediction))
    return result


def checkpoint(destination, name, combo, stage, parent_id, meta, features, fit, *, historical_run=None):
    prediction, probabilities, clocks, warnings = fit
    frame = meta[KEYS].reset_index(drop=True).assign(prediction=prediction)
    for i in LABELS:
        frame[f'p_{i}'] = probabilities[:, i]
    metrics = {**clocks, **metrics_for(frame), 'reused': historical_run is not None}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix='.writing-') as work:
        stage_dir = Path(work) / 'checkpoint'
        stage_dir.mkdir()
        frame.to_parquet(stage_dir / 'predictions.parquet', index=False, compression='zstd')
        np.savetxt(stage_dir / 'confusion_matrix.csv', confusion_matrix(frame.label, prediction, labels=LABELS),
                   fmt='%d', delimiter=',', header='pred_0,pred_1,pred_2,pred_3', comments='')
        write_json(stage_dir / 'features.json', features)
        write_json(stage_dir / 'warnings.json', warnings)
        with mlflow.start_run(run_name=f'{name}/{combo}/{stage}', nested=True) as run:
            mlflow.set_tags({'grid.batch': parent_id, 'feature_configuration': combo, 'model': name, 'stage': stage})
            if historical_run:
                mlflow.set_tags({'grid.reused': 'true', 'grid.source_run_id': str(historical_run)})
            mlflow.log_params({key: str(value) for key, value in pipeline(name).get_params().items()})
            mlflow.log_metrics({key: float(value) for key, value in metrics.items() if np.isfinite(value)})
            metrics['run_id'] = run.info.run_id
            metrics['predictions_sha256'] = sha256_file(stage_dir / 'predictions.parquet')
            write_json(stage_dir / 'metrics.json', metrics)
            mlflow.log_artifacts(str(stage_dir))
        stage_dir.rename(destination)
    print(f'{stage} {name}/{combo}: {metrics["critical_errors"]} critical, {clocks["training_seconds"]:.1f}s', flush=True)


def rank(frames):
    return sorted(frames, key=lambda key: (
        (frames[key].label.isin([1, 2]) & frames[key].prediction.eq(3)).sum(),
        (frames[key].label.eq(3) & frames[key].prediction.isin([1, 2])).sum(),
        key,
    ))


def completed_predictions(path):
    metrics = read_json(path / 'metrics.json')
    require_hash(path / 'predictions.parquet', metrics['predictions_sha256'])
    return pd.read_parquet(path / 'predictions.parquet')


def manual_archive(name, train):
    """Reuse native four-class predictions only after provenance and registry parity checks."""
    archive = EXPERIMENTS_ROOT / 'class2_group_cv_v1'
    verification = read_json(archive / 'verification.json')
    if verification['original_four_class_oof_exact_matches'] != 7:
        raise ValueError('Missing original four-class replay verification')
    for path, checksum in read_json(archive / 'input_hashes.json').items():
        parts = Path(path).parts
        start = next(i for i, part in enumerate(parts) if part in ('results', 'configs'))
        require_hash(PROJECT_ROOT.joinpath(*parts[start:]), checksum)
    path = archive / 'predictions' / f'four_class__{name}.csv'
    frame = pd.read_csv(path)
    pd.testing.assert_frame_equal(frame[KEYS], train[KEYS].reset_index(drop=True))
    probabilities_in_label_order(frame[[f'p_{i}' for i in LABELS]], LABELS)
    summaries = pd.read_csv(archive / 'fold_runs.csv').query("arm == 'four_class' and model == @name").set_index('fold')
    expected = {key: str(val) for key, val in pipeline(name).get_params(deep=True).items()}
    diagnostics = [read_json(archive/'models'/f'four_class__{name}__fold{i}.json') for i in range(3)]
    if summaries.index.tolist() != [0, 1, 2] or any(item['parameters'] != expected for item in diagnostics):
        raise ValueError('Historical fit configuration differs from registry')
    return frame, summaries, diagnostics, sha256_file(path)


def reuse_manual(output, name, train, grid, parent_id):
    destinations = [output/'fits'/'manual'/name/f'fold_{fold}' for fold in range(3)]
    if name == 'rbf_svm' or all(path.exists() for path in destinations):
        return
    try:
        frame, summaries, diagnostics, archive_sha = manual_archive(name, train)
        prepared = [grid.fold(fold, ['manual'])[2] for fold in range(3)]
        for features, diag in zip(prepared, diagnostics, strict=True):
            medians = features['preprocessing_by_set']['manual']['medians']
            if [f'manual::{name}' for name in medians] != features['retained_features']:
                raise ValueError('Manual feature order or retention differs')
            np.testing.assert_array_equal(list(medians.values()), diag['imputer_statistics'])
    except (OSError, ValueError, AssertionError, KeyError) as exc:
        print(f'Manual archive {name} unavailable or incompatible: {exc}; fitting normally', flush=True)
        return
    for fold, path in enumerate(destinations):
        if not path.exists():
            part = frame.loc[frame.fold.eq(fold)]
            source = summaries.loc[fold]
            features = {**prepared[fold], 'historical_oof_sha256': archive_sha,
                        'clock_note': 'archived combined fit and prediction; separate inference clock unavailable'}
            fit = (part.prediction.to_numpy(), part[[f'p_{i}' for i in LABELS]].to_numpy(),
                   {'training_seconds': float(source.seconds), 'prediction_seconds': 0.0}, [])
            checkpoint(path, name, 'manual', f'fold_{fold}', parent_id, part, features, fit,
                       historical_run=source.run_id)


def run_cv(output, grid, train, combos, parent_id, jobs, *, retry_failed):
    """Process every model phase across all combinations, with at most two fits in memory."""
    from src.analysis.feature_grid_report import write_report

    for name in MODELS:
        print(f'Model phase: {name}', flush=True)
        reuse_manual(output, name, train, grid, parent_id)
        pending = []

        def finish(item):
            destination, combo, fold, meta, features, future = item
            failure = output / 'failures' / combo / name / f'{fold}.json'
            try:
                checkpoint(destination, name, combo, fold, parent_id, meta, features, future.result())
            except Exception:
                failure.parent.mkdir(parents=True, exist_ok=True)
                write_json(failure, {'error': traceback.format_exc(), 'timestamp': time.time()})
                print(f'{fold} {name}/{combo}: failed, see {failure}', flush=True)

        width = 1 if name == 'rbf_svm' else jobs
        with ThreadPoolExecutor(max_workers=width) as pool:
            for combo, names in combos.items():
                for fold in range(3):
                    stage = f'fold_{fold}'
                    destination = output / 'fits' / combo / name / stage
                    failure = output / 'failures' / combo / name / f'{stage}.json'
                    if destination.exists() or (failure.exists() and not retry_failed):
                        continue
                    try:
                        X_train, X_val, features = grid.fold(fold, names)
                    except Exception:
                        failure.parent.mkdir(parents=True, exist_ok=True)
                        write_json(failure, {'error': traceback.format_exc(), 'timestamp': time.time()})
                        print(f'{stage} {name}/{combo}: feature preparation failed', flush=True)
                        continue
                    validation = train.loc[train.fold.eq(fold)]
                    y = train.loc[train.fold.ne(fold), 'label'].to_numpy()
                    future = pool.submit(fit_arrays, name, (X_train, y), X_val, output)
                    pending.append((destination, combo, stage, validation, features, future))
                    if len(pending) >= width:
                        finish(pending.pop(0))
            for item in pending:
                finish(item)
        write_report(output, combos, MODELS, train)


def run_holdouts(output, grid, membership, combos, parent_id):
    validation_plan = output / 'shortlist.json'
    test_plan = output / 'test_selection.json'
    if not validation_plan.exists():
        completed = {}
        for combo in combos:
            for model in MODELS:
                paths = [output/'fits'/combo/model/f'fold_{i}' for i in range(3)]
                if all(path.exists() for path in paths):
                    completed[(combo, model)] = pd.concat([completed_predictions(path) for path in paths])
        # A failed fold must not silently remove a candidate before shortlist selection.
        if len(completed) != len(combos) * len(MODELS):
            return
        write_json(validation_plan, {'source': 'pooled OOF only', 'pairs': rank(completed)[:SHORTLIST]})
    shortlist = [tuple(pair) for pair in read_json(validation_plan)['pairs']]
    frames = {}
    for split, candidates in [('validation', shortlist), ('test', None)]:
        if split == 'test':
            if not test_plan.exists():
                if len(frames) != len(shortlist):
                    return
                write_json(test_plan, {'source': 'validation shortlist, no test scores', 'pair': rank(frames)[0]})
            candidates = [tuple(read_json(test_plan)['pair'])]
        for combo, model in candidates:
            path = output / 'holdout' / combo / model / split
            if not path.exists():
                try:
                    X_train, X_val, X_test, features = grid.holdout(combos[combo])
                    train = membership.loc[membership.split.eq('train'), 'label'].to_numpy()
                    meta = membership.loc[membership.split.eq(split)].assign(fold=-1)
                    fit = fit_arrays(model, (X_train, train), X_val if split == 'validation' else X_test, output)
                    checkpoint(path, model, combo, split, parent_id, meta, features, fit)
                except Exception:
                    failure = output / 'failures' / combo / model / f'{split}.json'
                    failure.parent.mkdir(parents=True, exist_ok=True)
                    write_json(failure, {'error': traceback.format_exc(), 'timestamp': time.time()})
                    continue
            if split == 'validation':
                frames[(combo, model)] = completed_predictions(path)


def run(output, jobs, *, retry_failed=False, cv_only=False):
    from src.features.specimen_grid import GridFeatures

    output.mkdir(exist_ok=True, parents=True)
    with (output / '.runner.lock').open('a') as lock, threadpool_limits(1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = plan_config()
        plan_path = output / 'plan.json'
        membership, train = load_cohorts()
        grid = GridFeatures(membership, train, output / 'cache')
        plan['feature_inputs_sha256'] = grid.source_fingerprints
        if plan_path.exists():
            if read_json(plan_path) != plan:
                raise ValueError('Plan, data split or runtime changed; use a new output directory')
            # Parquet stores the ordered IDs and labels, but omits the pandas row index.
            pd.testing.assert_frame_equal(pd.read_parquet(output / 'cohort.parquet'), train.reset_index(drop=True))
        else:
            train.to_parquet(output / 'cohort.parquet', index=False)
            write_json(plan_path, plan)
        experiment = set_project_experiment(EXPERIMENT)
        mlflow.sklearn.autolog(disable=True)
        root = output / 'run.json'
        if root.exists():
            main = mlflow.start_run(run_id=read_json(root)['run_id'])
        else:
            main = mlflow.start_run(experiment_id=experiment.experiment_id, run_name=output.name)
        with main:
            parent_id = main.info.run_id
            if not root.exists():
                write_json(root, {'run_id': parent_id, 'experiment_id': experiment.experiment_id})
                mlflow.log_artifact(str(plan_path))
                mlflow.log_artifact(str(output / 'cohort.parquet'))
            combos = plan['combinations']
            run_cv(output, grid, train, combos, parent_id, jobs, retry_failed=retry_failed)
            if not cv_only:
                run_holdouts(output, grid, membership, combos, parent_id)
            from src.analysis.feature_grid_report import write_report

            write_report(output, combos, MODELS, train)
            mlflow.log_artifacts(str(output / 'report'), 'report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, choices=[1, 2], default=2)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--cv-only', action='store_true')
    args = parser.parse_args()
    run(args.output_dir, args.jobs, retry_failed=args.retry_failed, cv_only=args.cv_only)


if __name__ == '__main__':
    main()
