"""Run a four-class comparison of the fused feature set on frozen specimen folds."""
import argparse
import importlib.metadata
import os
import platform
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.split_manifest import SPLIT_NAMES, restore_splits, sha256_file
from src.evaluation.metrics import ERROR_METRICS_VERSION
from src.evaluation.reporting import evaluate_holdout
from src.experiments.model_registry import MODEL_ENABLED, build_models
from src.experiments.provenance import capture_git_state, snapshot_source_tree
from src.experiments.tracking import set_project_experiment
from src.experiments.training import fit_and_predict, shared_rbf_candidates
from src.features.extract_tsfresh_features import select_training_features
from src.features.specimen_fusion import (
    HOLDOUT_HASHES,
    fuse_features,
    load_manual_holdout,
)
from src.features.specimen_inputs import (
    ANCHOR,
    EARLY,
    FOLDS,
    SPLITS,
    TSFRESH,
    load_cohorts,
    load_early_manual_train,
    load_tsfresh_holdout,
    raw_tsfresh,
    read_json,
    require_hash,
    tsfresh_manifest,
    write_json,
)
from src.paths import FEATURE_TABLES_ROOT, PROJECT_ROOT

SEED = 42
FEATURE_SET = 'specimen_manual52_early_tsfresh_fused_v1'
SVM_PARAMS = {'model__C': 0.3, 'model__gamma': 0.001, 'model__class_weight': 'balanced'}
EXPERIMENT = 'thz-specimen-feature-fusion'


def model_pipelines():
    models = build_models(enabled=dict(MODEL_ENABLED, rbf_svm=True), seed=SEED)
    models['rbf_svm'].set_params(model__cache_size=768, **SVM_PARAMS)
    return models


def metadata_plan(feature_file):
    inputs = [
        SPLITS / 'manifest.json', FOLDS / 'manifest.json', ANCHOR / 'extraction_manifest.json',
        ANCHOR / 'features' / f'{EARLY}.npz', TSFRESH / 'manifest.json', TSFRESH / 'imputation.csv',
    ]
    return {
        'feature_set': FEATURE_SET, 'feature_file': str(feature_file.resolve()),
        'seed': SEED, 'metrics_version': ERROR_METRICS_VERSION, 'threads_per_fit': 1,
        'parameters': {name: {k: str(v) for k, v in model.get_params().items()}
                       for name, model in model_pipelines().items()},
        'rbf_cv': 'exact precomputed RBF, 192 MiB solver cache',
        'classes': [0, 1, 2, 3], 'holdout_role': 'evaluation only',
        'fusion_order': ['manual52_early', 'tsfresh'],
        'filter_scope': 'fold training or original training, exact column equality',
        'source_hashes': {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in inputs},
        'early_manual_holdout_hashes': HOLDOUT_HASHES,
        'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()},
        'git': capture_git_state(PROJECT_ROOT),
    }


def write_fused_table(path, output, membership, train, manual, tsfresh):
    """Persist the final-fit feature table and its input column lineage."""
    data, state = fuse_features(load_manual_holdout(membership, manual), tsfresh, train.index)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = path.with_suffix('.json')
    if path.exists() or manifest_path.exists():
        raise FileExistsError(path)
    frame = pd.concat([membership[['instance_id', 'label']], data], axis=1)
    with path.open('xb') as stream:
        frame.to_parquet(stream, engine='pyarrow', compression='zstd', index=False)
    saved = pd.read_parquet(path)
    pd.testing.assert_frame_equal(frame, saved, check_exact=True)
    state.update({'file': path.name, 'sha256': sha256_file(path), 'points': len(data),
                  'training_points': len(train), 'feature_set': FEATURE_SET})
    write_json(manifest_path, state)
    shutil.copyfile(manifest_path, output / 'final_features.json')
    shutil.copyfile(TSFRESH / 'imputation.csv', output / 'final_tsfresh_imputation.csv')
    return data


def fit_evaluate(name, model, X, y, evaluations, output, *, precompute_rbf=False):
    """Common fit interface, with disk-backed RBF CV only for large dense matrices."""
    if precompute_rbf:
        with ThreadPoolExecutor(max_workers=1) as pool:
            (_, prediction, timing), = shared_rbf_candidates(
                X, evaluations['cv'], np.asarray(y), pool, output, grid=[SVM_PARAMS], seed=SEED,
            )
        return {'cv': prediction}, timing.pop('training_seconds'), {
            'cv': timing.pop('prediction_seconds'),
        }, timing
    predictions, fit, durations = fit_and_predict(model, X, y, evaluations)
    extras = {'support_vectors': int(model[-1].n_support_.sum())} if name == 'rbf_svm' else {}
    return predictions, fit, durations, extras


def run_model(name, model, training, evaluations, metadata, root, phase, parent_id):
    """Fit, evaluate, and log a single registry pipeline using shared reports."""
    destination = root / phase / name
    destination.mkdir(parents=True)
    params = {key: str(value) for key, value in model.get_params(deep=True).items()}
    with mlflow.start_run(run_name=f'{phase}-{name}', nested=True) as run:
        mlflow.log_params(params)
        mlflow.set_tags({'feature_set': FEATURE_SET, 'model': name, 'phase': phase, 'batch': parent_id})
        X, y = training
        accelerated = phase.startswith('fold_') and name == 'rbf_svm'
        mlflow.log_param('fit_backend', 'precomputed_rbf' if accelerated else 'registry_pipeline')
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pred, fit, durations, extra = fit_evaluate(
                name, model, X, y, evaluations, root, precompute_rbf=accelerated,
            )
        wall = time.perf_counter() - started
        write_json(destination / 'warnings.json', [str(w.message) for w in caught])
        mlflow.log_artifact(str(destination / 'warnings.json'))
        mlflow.log_metrics({
            'training_seconds': fit, 'fit_and_predict_wall_seconds': wall,
            'input_features': X.shape[1], 'training_points': len(X), **extra,
        })
        if phase == 'holdout':
            model_file = destination / 'model.joblib'
            joblib.dump(model, model_file, compress=3)
            saved = joblib.load(model_file)
            for split, X_split in evaluations.items():
                sample = np.linspace(0, len(X_split) - 1, 64, dtype=int)
                np.testing.assert_array_equal(saved.predict(X_split[sample]), pred[split][sample])
            mlflow.log_artifact(str(model_file))
            mlflow.set_tag('reload_check', '64 evenly spaced predictions per split matched')
        rows = []
        for split, prediction in pred.items():
            part = metadata[split]
            metrics = evaluate_holdout(part, prediction, destination / split, include_critical_breakdown=True)
            mlflow.log_metrics({f'{split}_{key}': value for key, value in metrics.items()})
            mlflow.log_metric(f'{split}_prediction_seconds', durations[split])
            mlflow.log_artifacts(str(destination / split), artifact_path=split)
            rows.append(dict(
                model=name, phase=phase, split=split, run_id=run.info.run_id,
                training_seconds=fit, prediction_seconds=durations[split], fit_predict_wall_seconds=wall,
                input_features=X.shape[1], **extra, **metrics,
            ))
        print(f'{phase} {name}: fit {fit:.1f}s, critical {[row["critical_errors"] for row in rows]}', flush=True)
    return pred, rows


def run(output, feature_file):
    if feature_file.suffix != '.parquet':
        raise ValueError('The fused feature file must end in .parquet')
    if output.exists() or feature_file.exists() or feature_file.with_suffix('.json').exists():
        raise FileExistsError('Use a fresh experiment directory and feature-file path')
    if mlflow.active_run() is not None or os.environ.get('MLFLOW_RUN_ID'):
        raise ValueError('Fresh MLflow runs required')

    started = time.perf_counter()
    membership, train = load_cohorts()
    manifest = tsfresh_manifest()
    manual = load_early_manual_train(train)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', metadata_plan(feature_file))
    snapshot_source_tree(PROJECT_ROOT / 'src', output / 'source')
    set_project_experiment(EXPERIMENT)
    mlflow.sklearn.autolog(disable=True)
    with threadpool_limits(1), mlflow.start_run(run_name=FEATURE_SET) as parent:
        parent_id = parent.info.run_id
        write_json(output / 'run.json', {'experiment_id': parent.info.experiment_id, 'run_id': parent_id})
        mlflow.log_artifact(str(output / 'plan.json'))
        mlflow.log_artifacts(str(output / 'source'), artifact_path='source')
        mlflow.log_params({'seed': SEED, 'feature_set': FEATURE_SET, 'error_metrics': ERROR_METRICS_VERSION})
        prep_start = time.perf_counter()
        # The final table uses preprocessing fitted on all training specimens.
        # Group CV rebuilds the features from separate raw inputs below.
        final = write_fused_table(feature_file, output, membership, train,
                                  manual, load_tsfresh_holdout(membership, manifest))
        mlflow.log_artifact(str(output / 'final_features.json'))
        mlflow.log_artifact(str(output / 'final_tsfresh_imputation.csv'))
        mlflow.log_metric('feature_table_preparation_seconds', time.perf_counter() - prep_start)
        mlflow.log_metric('final_input_features', final.shape[1])
        print(f'Fused table saved: {final.shape}', flush=True)
        del final

        raw = raw_tsfresh(membership, train, manifest)
        predictions = {name: np.full(len(train), -1) for name in model_pipelines()}
        cv_rows = []
        cv_start = time.perf_counter()
        for fold in sorted(train.fold.unique()):
            mask = train.fold.eq(fold)
            fit_ids = train.index[~mask]
            phase = f'fold_{fold}'
            prep_start = time.perf_counter()
            # Fit both tsfresh preprocessing and cross-block deduplication on
            # the same fold-training IDs so held-out specimens cannot select columns.
            selected, ranges = select_training_features(raw, train.label, fit_ids, return_ranges=True)
            matrix, state = fuse_features(manual, selected, fit_ids)
            prep_seconds = time.perf_counter() - prep_start
            write_json(output / f'{phase}_features.json', state)
            ranges_file = output / f'{phase}_tsfresh_imputation.csv'
            pd.DataFrame(dict(zip(('max', 'min', 'median'), ranges))).to_csv(ranges_file)
            mlflow.log_artifact(str(output / f'{phase}_features.json'))
            mlflow.log_artifact(str(ranges_file))
            mlflow.log_metric(f'{phase}_preparation_seconds', prep_seconds)
            print(f'{phase}: {matrix.shape[1]} fused features, prep {prep_seconds:.1f}s', flush=True)
            X_train = np.ascontiguousarray(matrix.loc[~mask], dtype=np.float64)
            X_val = np.ascontiguousarray(matrix.loc[mask], dtype=np.float64)
            for name, model in model_pipelines().items():
                pred, rows = run_model(name, model, (X_train, train.label.loc[~mask]), {'cv': X_val},
                                       {'cv': train.loc[mask]}, output, phase, parent_id)
                predictions[name][mask.to_numpy()] = pred['cv']
                cv_rows.extend(rows)
                pd.DataFrame(cv_rows).to_csv(output / 'fold_metrics.csv', index=False)
        del raw, selected, matrix, X_train, X_val
        mlflow.log_metric('cv_wall_seconds', time.perf_counter() - cv_start)
        pooled = []
        for name, prediction in predictions.items():
            if not np.isin(prediction, [0, 1, 2, 3]).all():
                raise ValueError('Missing out-of-fold predictions')
            metrics = evaluate_holdout(train, prediction, output / 'oof' / name, include_critical_breakdown=True)
            pooled.append({'model': name, **metrics})
        pd.DataFrame(pooled).sort_values(['critical_errors', 'false_alarms', 'model']).to_csv(
            output / 'pooled_oof_metrics.csv', index=False,
        )
        mlflow.log_artifacts(str(output / 'oof'), artifact_path='oof')
        mlflow.log_artifact(str(output / 'fold_metrics.csv'))
        mlflow.log_artifact(str(output / 'pooled_oof_metrics.csv'))

        require_hash(feature_file, read_json(output / 'final_features.json')['sha256'])
        table = pd.read_parquet(feature_file)
        frames = restore_splits(table.drop(columns=['instance_id', 'label']), table.label, table.instance_id, SPLITS)
        X_train, X_val, X_test = [np.ascontiguousarray(frame, dtype=np.float64) for frame in frames[:3]]
        outcomes = {'validation': X_val, 'test': X_test}
        metadata = {split: membership.loc[frame.index] for split, frame in zip(SPLIT_NAMES[1:], frames[1:3])}
        holdout_rows = []
        for name, model in model_pipelines().items():
            _, rows = run_model(name, model, (X_train, frames[3]), outcomes, metadata, output, 'holdout', parent_id)
            holdout_rows.extend(rows)
            pd.DataFrame(holdout_rows).to_csv(output / 'holdout_metrics.csv', index=False)
        mlflow.log_artifact(str(output / 'holdout_metrics.csv'))
        total = time.perf_counter() - started
        write_json(output / 'runtime.json', {'wall_seconds': total, 'run_id': parent_id})
        mlflow.log_metric('wall_seconds', total)
        mlflow.log_artifact(str(output / 'runtime.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--feature-file', type=Path, default=FEATURE_TABLES_ROOT / f'{FEATURE_SET}.parquet')
    args = parser.parse_args()
    run(args.output_dir, args.feature_file)


if __name__ == '__main__':
    main()
