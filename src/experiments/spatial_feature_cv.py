"""Compare point and spatial input features in the frozen nested SVM design."""

import argparse
import fcntl
import json
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from platform import python_version

import mlflow
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.data.split_manifest import sha256_file
from src.evaluation.svm_nested_selection import BASELINE_TOLERANCE, GUARDRAILS, choose_policy, raw_metrics
from src.experiments.spatial_feature_inputs import load_spatial_feature_inputs, load_spatial_feature_reference
from src.experiments.svm_nested_hpo import (
    FEATURES,
    KEYS,
    SEED,
    atomic_json,
    atomic_parquet,
    candidate_grid,
    fit_checkpoint,
    prepare_fold,
    read_fit,
)
from src.experiments.svm_nested_kernel import iter_rbf_fits
from src.features.spatial_context import SIGNAL_CHANNELS, SpatialContext
from src.features.specimen_inputs import read_json
from src.paths import EXPERIMENTS_ROOT, PROJECT_ROOT

NESTED = EXPERIMENTS_ROOT / 'specimen_svm_nested_hpo_v1'
POST = EXPERIMENTS_ROOT / 'specimen_spatial_postprocessing_v1_verified'
OUTPUT = EXPERIMENTS_ROOT / 'specimen_spatial_features_v1'
ARMS = {'tight': 1.5, 'wide': 4.0}
STATS = ('delta', 'std')
METADATA = [*KEYS, 'x', 'y']


def augment_features(point_fit, point_eval, spatial, fit_index, eval_index):
    """Fit context medians, variance filtering and scaling only on fit rows."""
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    fit = imputer.fit_transform(spatial.loc[fit_index])
    evaluation = imputer.transform(spatial.loc[eval_index])
    keep = np.any(fit != fit[0], axis=0)
    if not keep.any():
        return point_fit, point_eval, {'columns': [], 'width_factor': 1.}

    scaler = StandardScaler()
    scaled_fit = scaler.fit_transform(fit[:, keep])
    scaled_eval = scaler.transform(evaluation[:, keep])
    # Preserve the reference kernel's expected squared distance for standardized rows.
    width = point_fit.shape[1]
    factor = float(np.sqrt(width / (width + int(keep.sum()))))
    return (
        np.concatenate((point_fit, scaled_fit), axis=1) * factor,
        np.concatenate((point_eval, scaled_eval), axis=1) * factor,
        {'columns': spatial.columns[keep].tolist(), 'median': imputer.statistics_[keep].tolist(),
         'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'width_factor': factor},
    )


def fit_arm(directory, fit, evaluation, point_fit, point_eval, spatial, candidate, workspace):
    if not directory.exists():
        X, Z, state = augment_features(point_fit, point_eval, spatial, fit.index, evaluation.index)
        result, = iter_rbf_fits(X, fit.label.to_numpy(), Z, [candidate], workspace, seed=SEED)
        _, prediction, scores, clocks = result
        fit_checkpoint(directory, evaluation, prediction, scores, {**clocks, 'preprocessing': state})
    frame, _ = read_fit(directory, evaluation)
    return frame


def select_arm(inner_predictions):
    """Choose a single optional spatial scale from inner OOF with the nested safety policy."""
    trials = pd.DataFrame([{'arm': arm, **raw_metrics(predictions)}
                           for arm, predictions in inner_predictions.items()])
    control = trials.loc[trials.arm.eq('point_only')].iloc[0].to_dict()
    choice = choose_policy(trials, control)
    return choice, trials


def checkpoint_paths(nested, inputs):
    """Freeze the archived point-only controls and the inner-selected SVM setup."""
    controls, hashes = {}, {}
    candidates = {item['candidate']: item for item in candidate_grid()}
    for feature_set in FEATURES:
        for outer in sorted(inputs.train.fold.unique()):
            work = nested / 'search' / feature_set / f'outer_{outer}'
            path = work / 'selection.json'
            hashes[str(path)] = sha256_file(path)
            regular = read_json(path)['regular']
            candidate = candidates[regular['candidate']]
            provenance = nested / 'outer' / feature_set / f'fold_{outer}' / 'provenance.json'
            hashes[str(provenance)] = sha256_file(provenance)
            for directory in [*(work / f'inner_{k}' / candidate['candidate'] for k in range(3)),
                              work / 'outer_fits' / candidate['candidate']]:
                for name in ('fit.json', 'predictions.parquet'):
                    source = directory / name
                    hashes[str(source)] = sha256_file(source)
            controls[feature_set, int(outer)] = (work, candidate)
    return controls, hashes


def run_fold(output, inputs, feature_set, outer, work, candidate, spatial, pool):
    train = inputs.train
    assignments = inputs.assignments.loc[inputs.assignments.outer.eq(outer)]
    directory = output / 'fits' / feature_set / f'fold_{outer}'
    inner_rows = {arm: [] for arm in ('point_only', *ARMS)}
    for inner in sorted(assignments.fold.unique()):
        fit, evaluation = prepare_fold(train, assignments, inner)
        stage = directory / f'inner_{inner}'
        archive = work / f'inner_{inner}' / candidate['candidate']
        inner_rows['point_only'].append(read_fit(archive, evaluation)[0])
        point_fit, point_eval, _ = inputs.features.prepare(fit, evaluation, FEATURES[feature_set])
        pending = {
            arm: pool.submit(fit_arm, stage / arm, fit, evaluation, point_fit, point_eval,
                             spatial[arm], candidate, output)
            for arm in ARMS
        }
        for arm, future in pending.items():
            inner_rows[arm].append(future.result())
        print(f'{feature_set} outer={outer} inner={inner}: both radii complete', flush=True)

    choice, trials = select_arm({key: pd.concat(rows, ignore_index=True) for key, rows in inner_rows.items()})
    # Save the scale choice before assessing the outer fold.
    record = directory / 'selection.json'
    if record.exists() and choice != read_json(record):
        raise ValueError(f'Changed inner selection in {directory}')
    if not record.exists():
        atomic_json(record, choice)

    fit, assessment = train.loc[train.fold.ne(outer)], train.loc[train.fold.eq(outer)]
    point_fit, point_eval, _ = inputs.features.prepare(fit, assessment, FEATURES[feature_set])
    archive = work / 'outer_fits' / candidate['candidate']
    reference, _ = read_fit(archive, assessment)
    predictions = {'point_only': reference.prediction.to_numpy()}
    for arm in ARMS:
        # The outer pair of dense kernels can exceed 6 GB per arm. Fit serially here.
        assessed = fit_arm(directory / 'outer' / arm, fit, assessment, point_fit, point_eval,
                           spatial[arm], candidate, output)
        predictions[arm] = assessed.prediction.to_numpy()
    predictions['selected'] = predictions[choice['arm']]
    metadata = assessment[METADATA].reset_index(drop=True).assign(feature_set=feature_set)
    result = pd.concat([metadata.assign(arm=arm, prediction=labels) for arm, labels in predictions.items()],
                       ignore_index=True)
    print(f'{feature_set} outer={outer}: assessed with selection {choice["arm"]}', flush=True)
    return result, trials.assign(feature_set=feature_set, outer_fold=outer, candidate=candidate['candidate'],
                                 selected=trials.arm.eq(choice['arm']))


def track(output, report):
    """Log in the run's own SQLite store and retain the historical store byte-for-byte."""
    client = mlflow.MlflowClient(tracking_uri=f'sqlite:///{output / "mlflow.db"}')
    experiment = client.create_experiment('thz-spatial-svm-inputs', artifact_location=(output / 'mlartifacts').as_uri())
    run = client.create_run(experiment, tags={'mlflow.runName': output.name, 'scope': 'train_only_nested_oof'})
    run_id = run.info.run_id
    try:
        for name, val in {'seed': SEED, 'outer_folds': 3, 'inner_folds': 3, 'radii': list(ARMS.values()),
                          'spatial_statistics': STATS, 'model': 'exact RBF SVC'}.items():
            client.log_param(run_id, name, val)
        for row in report['pooled'].itertuples(index=False):
            key = f'{row.feature_set.replace("+", "__")}.{row.arm}'
            for metric in ('critical_error_rate', 'cavity_miss_rate', 'uncertain_miss_rate',
                           'false_alarm_rate', 'macro_f1', 'balanced_accuracy', 'release_rate'):
                client.log_metric(run_id, f'{key}.{metric}', getattr(row, metric))
        for name in ('plan.json', 'reference_hashes.json', 'inner_trials.csv', 'predictions.parquet'):
            client.log_artifact(run_id, str(output / name))
        client.log_artifacts(run_id, str(output / 'report'), 'report')
    except Exception:
        client.set_terminated(run_id, 'FAILED')
        raise
    client.set_terminated(run_id)
    atomic_json(output / 'run.json', {'run_id': run_id, 'experiment_id': experiment, 'tracking': 'mlflow.db'})


def run(nested, post, output, workers, *, resume=False):
    nested, post, output = [Path(path).resolve() for path in (nested, post, output)]
    if any(output.is_relative_to(source) for source in (nested, post)):
        raise ValueError('Write outside historical experiment directories')
    if workers not in (1, 2):
        raise ValueError('Use one or two workers for the disk-backed kernels')
    if resume and not (output / 'plan.json').is_file():
        raise FileNotFoundError('Resume requires an existing frozen run plan')
    output.mkdir(parents=True, exist_ok=resume)
    with (output / '.run.lock').open('a') as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'run.json').exists():
            raise FileExistsError('Experiment already finished')
        inputs = load_spatial_feature_inputs(nested)
        controls, hashes = checkpoint_paths(nested, inputs)
        source_paths = ('src/experiments/spatial_feature_cv.py', 'src/experiments/spatial_feature_inputs.py',
                        'src/features/spatial_context.py', 'src/evaluation/spatial_postprocessing.py',
                        'src/experiments/svm_nested_hpo.py', 'src/experiments/svm_nested_kernel.py',
                        'src/evaluation/svm_nested_selection.py', 'src/features/specimen_grid.py')
        plan = {
            'protocol': 'spatial_input_frozen_rbf_v1', 'seed': SEED, 'nested': str(nested), 'post': str(post),
            'arms': ARMS, 'statistics': list(STATS), 'signal_channels': list(SIGNAL_CHANNELS),
            'weighting': 'gaussian', 'feature_sets': list(FEATURES),
            'capacity': 'point-only inner-selected C/gamma/weights frozen for both spatial arms',
            'kernel_width': 'sqrt(point width / concatenated width) on standardized vectors',
            'selection': {'guardrails': GUARDRAILS, 'baseline_tolerance': BASELINE_TOLERANCE},
            'point_control': {f'{features}/{outer}': candidate['candidate']
                              for (features, outer), (_, candidate) in controls.items()},
            'inputs_sha256': {**inputs.hashes, **hashes},
            'code_sha256': {path: sha256_file(PROJECT_ROOT / path) for path in source_paths},
            'python': python_version(),
            'packages': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn',
                                                           'scipy', 'pyarrow', 'mlflow')},
        }
        path = output / 'plan.json'
        if path.exists() and read_json(path) != json.loads(json.dumps(plan)):
            raise ValueError('Code or archived inputs changed since the experiment started')
        if not path.exists():
            atomic_json(path, plan)

        # No pooled statistics across bodies, labels or predictions enter the spatial transform.
        geometry = SpatialContext(inputs.train[['specimen_id', 'source_dataset', 'x', 'y']], radii=tuple(ARMS.values()))
        channels = inputs.features.frames['manual'].loc[:, list(SIGNAL_CHANNELS)]
        spatial = {arm: geometry.transform(channels, radii=(radius,), statistics=STATS)
                   for arm, radius in ARMS.items()}
        frames, trials = [], []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for (feature_set, fold), (work, candidate) in controls.items():
                result, inner = run_fold(output, inputs, feature_set, fold, work, candidate, spatial, pool)
                frames.append(result)
                trials.append(inner)

        # Inspect prior postprocessing decisions only after all scale choices are fixed.
        previous, previous_hashes = load_spatial_feature_reference(nested, post, inputs.train, tuple(FEATURES))
        predictions = pd.concat([*frames, previous], ignore_index=True)
        for names in ('point_only', 'prior_native'):
            rows = predictions.loc[predictions.arm.eq(names)].drop(columns='arm')
            if names == 'point_only':
                control = rows.sort_values(['feature_set', 'instance_id'], ignore_index=True)
            else:
                pd.testing.assert_frame_equal(control, rows[control.columns].sort_values(
                    ['feature_set', 'instance_id'], ignore_index=True), check_dtype=False)

        from src.analysis.spatial_feature_report import summarize_spatial_features

        report = summarize_spatial_features(predictions)
        atomic_json(output / 'reference_hashes.json', previous_hashes)
        atomic_parquet(output / 'predictions.parquet', predictions)
        pd.concat(trials, ignore_index=True).to_csv(output / 'inner_trials.csv', index=False)
        directory = output / 'report'
        directory.mkdir(exist_ok=True)
        for name, table in report.items():
            table.to_csv(directory / f'{name}.csv', index=False)
        track(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nested', type=Path, default=NESTED)
    parser.add_argument('--post', type=Path, default=POST)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.nested, args.post, args.output_dir, args.workers, resume=args.resume)


if __name__ == '__main__':
    main()
