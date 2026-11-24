"""Retune the RBF SVM for a radius screened on frozen inner-fold predictions."""

import argparse
import fcntl
import json
from importlib.metadata import version
from pathlib import Path
from platform import python_version

import mlflow
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.split_manifest import sha256_file
from src.evaluation.svm_nested_selection import BASELINE_TOLERANCE, GUARDRAILS, choose_policy, raw_metrics
from src.experiments.spatial_feature_cv import ARMS, METADATA, NESTED, OUTPUT as FROZEN, STATS, augment_features
from src.experiments.spatial_feature_inputs import load_spatial_feature_inputs
from src.experiments.svm_nested_hpo import (
    FEATURES, SEED, atomic_json, atomic_parquet, candidate_grid, fit_checkpoint, prepare_fold, read_fit,
)
from src.experiments.svm_nested_kernel import iter_rbf_fits
from src.features.spatial_context import SIGNAL_CHANNELS, SpatialContext
from src.features.specimen_inputs import read_json
from src.paths import EXPERIMENTS_ROOT, PROJECT_ROOT

OUTPUT = EXPERIMENTS_ROOT / 'specimen_spatial_features_hpo_v1'
EXPERIMENT = 'thz-spatial-svm-input-hpo'
POINT = 'point_only'
PROTOCOL = 'spatial_input_nested_rbf_v1'


def _fit_hashes(directory):
    return {str(path): sha256_file(path)
            for path in (directory / 'fit.json', directory / 'predictions.parquet')}


def _assert_or_write(path, value):
    if not path.exists():
        atomic_json(path, value)
    elif read_json(path) != json.loads(json.dumps(value)):
        raise ValueError(f'Frozen HPO state changed: {path}')


def screen_radius(metrics):
    """Screen only spatial arms, using the original point fit as the quality reference."""
    trials = pd.DataFrame([{'arm': arm, **metrics[arm]} for arm in ARMS])
    return choose_policy(trials, metrics[POINT])


def search_choices(trials, point):
    """Return the best spatial fit and the best deployable inner choice."""
    spatial = choose_policy(trials, point)
    choices = pd.concat([pd.DataFrame([point]), trials], ignore_index=True)
    return spatial, choose_policy(choices, point)


def freeze_plan(output, nested, frozen, inputs, feature_sets):
    baseline_path = frozen / 'plan.json'
    baseline = read_json(baseline_path)
    if (baseline['protocol'] != 'spatial_input_frozen_rbf_v1' or baseline['arms'] != ARMS
            or baseline['statistics'] != list(STATS) or baseline['signal_channels'] != list(SIGNAL_CHANNELS)
            or any(baseline['inputs_sha256'].get(path) != digest for path, digest in inputs.hashes.items())):
        raise ValueError('Frozen spatial experiment does not match current inputs or arms')
    paths = {str(baseline_path): sha256_file(baseline_path)}
    point_candidates = {}
    for name in feature_sets:
        for outer in sorted(inputs.train.fold.unique()):
            work = nested / 'search' / name / f'outer_{outer}'
            selection_path = work / 'selection.json'
            for inner in range(3):
                for arm in ARMS:
                    archive_fit = frozen / 'fits' / name / f'fold_{outer}' / f'inner_{inner}' / arm
                    paths.update(_fit_hashes(archive_fit))
            point_candidate = read_json(selection_path)['regular']['candidate']
            key = f'{name}/{outer}'
            if point_candidate != baseline['point_control'][key]:
                raise ValueError(f'Point configuration changed: {key}')
            point_candidates[key] = point_candidate
            point_hashes = {str(selection_path): sha256_file(selection_path)}
            for inner in range(3):
                point_hashes.update(_fit_hashes(work / f'inner_{inner}' / point_candidate))
            point_hashes.update(_fit_hashes(work / 'outer_fits' / point_candidate))
            if any(baseline['inputs_sha256'].get(path) != digest for path, digest in point_hashes.items()):
                raise ValueError(f'Point checkpoint inputs changed: {key}')
            paths.update(point_hashes)

    sources = [
        'src/experiments/spatial_feature_hpo.py', 'src/experiments/spatial_feature_cv.py',
        'src/experiments/spatial_feature_inputs.py', 'src/experiments/spatial_inputs.py',
        'src/experiments/svm_nested_hpo.py', 'src/experiments/svm_nested_kernel.py',
        'src/experiments/model_registry.py', 'src/evaluation/svm_nested_selection.py',
        'src/features/spatial_context.py', 'src/features/specimen_grid.py', 'src/features/specimen_inputs.py',
    ]
    packages = {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'scipy', 'pyarrow', 'mlflow')}
    if packages != baseline['packages'] or python_version() != baseline['python']:
        raise ValueError('Runtime differs from the frozen spatial experiment')
    code = {path: sha256_file(PROJECT_ROOT / path) for path in sources}
    for path in ('src/experiments/svm_nested_kernel.py', 'src/features/spatial_context.py',
                 'src/evaluation/svm_nested_selection.py', 'src/features/specimen_grid.py'):
        if code[path] != baseline['code_sha256'][path]:
            raise ValueError(f'Frozen spatial calculation changed: {path}')
    plan = {
        'protocol': PROTOCOL, 'seed': SEED, 'nested': str(nested), 'frozen_spatial': str(frozen),
        'feature_sets': list(feature_sets), 'arms': ARMS, 'statistics': list(STATS),
        'signal_channels': list(SIGNAL_CHANNELS), 'candidates': candidate_grid(),
        'screen': 'best spatial radius at frozen point parameters on each outer training block',
        'selection': {'guardrails': GUARDRAILS, 'baseline_tolerance': BASELINE_TOLERANCE},
        'point_control': point_candidates, 'inputs_sha256': inputs.hashes | paths,
        'code_sha256': code, 'python': python_version(), 'packages': packages,
    }
    _assert_or_write(output / 'plan.json', plan)
    return point_candidates


def _prepare(inputs, feature_set, fit, evaluation, spatial, recorded):
    point_fit, point_eval, _ = inputs.features.prepare(fit, evaluation, FEATURES[feature_set])
    X, Z, state = augment_features(point_fit, point_eval, spatial, fit.index, evaluation.index)
    expected = recorded['preprocessing']
    if state.keys() != expected.keys() or state['columns'] != expected['columns']:
        raise ValueError('Spatial preprocessing columns differ from the frozen checkpoint')
    for key in state.keys() - {'columns'}:
        np.testing.assert_allclose(state[key], expected[key], rtol=1e-12, atol=1e-13,
                                   err_msg=f'Spatial preprocessing {key} differs from the frozen checkpoint')
    return X, Z, state


def run_fold(output, nested, frozen, inputs, feature_set, outer, spatial, point_candidate, jobs):
    train = inputs.train
    assignments = inputs.assignments.loc[inputs.assignments.outer.eq(outer)]
    work = output / 'fits' / feature_set / f'fold_{outer}'
    work.mkdir(parents=True, exist_ok=True)
    archive = frozen / 'fits' / feature_set / f'fold_{outer}'
    point = nested / 'search' / feature_set / f'outer_{outer}'
    splits = [prepare_fold(train, assignments, inner) for inner in range(3)]
    controls = {
        arm: [read_fit((point / f'inner_{inner}' / point_candidate) if arm == POINT else
                       (archive / f'inner_{inner}' / arm), evaluation)
              for inner, (_, evaluation) in enumerate(splits)]
        for arm in (POINT, *ARMS)
    }
    screen_metrics = {arm: raw_metrics(pd.concat([frame for frame, _ in fits], ignore_index=True))
                      for arm, fits in controls.items()}
    screen = screen_radius(screen_metrics)
    radius = screen['arm']
    _assert_or_write(work / 'radius.json', {'selection': screen, 'trials': screen_metrics})
    print(f'{feature_set} outer={outer}: screen {radius} at {point_candidate}', flush=True)

    candidates = candidate_grid()
    candidates.sort(key=lambda candidate: candidate['candidate'] != point_candidate)
    spatial_frames = {candidate['candidate']: [] for candidate in candidates}
    for inner, (fit, evaluation) in enumerate(splits):
        stage = work / f'inner_{inner}'
        remaining = [candidate for candidate in candidates if candidate['candidate'] != point_candidate
                     and not (stage / candidate['candidate']).exists()]
        cached, preprocessing = controls[radius][inner]
        if remaining:
            X, Z, state = _prepare(inputs, feature_set, fit, evaluation, spatial[radius], preprocessing)
            print(f'{feature_set} outer={outer} inner={inner}: fitting {len(remaining)} configurations', flush=True)
            for candidate, prediction, scores, clocks in iter_rbf_fits(
                X, fit.label.to_numpy(), Z, remaining, output, seed=SEED, jobs=jobs,
            ):
                fit_checkpoint(stage / candidate['candidate'], evaluation, prediction, scores,
                               {**clocks, 'preprocessing': state})
        for candidate in candidates:
            key = candidate['candidate']
            spatial_frames[key].append(cached if key == point_candidate else read_fit(stage / key, evaluation)[0])
        print(f'{feature_set} outer={outer} inner={inner}: all {len(candidates)} configurations verified', flush=True)

    trials = pd.DataFrame([{'arm': radius, 'candidate': key, **raw_metrics(pd.concat(frames, ignore_index=True))}
                           for key, frames in spatial_frames.items()])
    point_trial = {'arm': POINT, 'candidate': point_candidate, **screen_metrics[POINT]}
    tuned, selected = search_choices(trials, point_trial)
    selection = {'tuned': tuned, 'selected': selected, 'point_candidate': point_candidate}
    _assert_or_write(work / 'selection.json', selection)

    fit, evaluation = train.loc[train.fold.ne(outer)], train.loc[train.fold.eq(outer)]
    frozen_dir = archive / 'outer' / radius
    frozen_frame, preprocessing = read_fit(frozen_dir, evaluation)
    _assert_or_write(work / 'frozen_outer.json', _fit_hashes(frozen_dir))
    if tuned['candidate'] == point_candidate:
        tuned_frame = frozen_frame
    else:
        stage = work / 'outer' / tuned['candidate']
        if not stage.exists():
            candidate = next(item for item in candidates if item['candidate'] == tuned['candidate'])
            X, Z, state = _prepare(inputs, feature_set, fit, evaluation, spatial[radius], preprocessing)
            result, = iter_rbf_fits(X, fit.label.to_numpy(), Z, [candidate], output, seed=SEED, jobs=jobs)
            _, prediction, scores, clocks = result
            fit_checkpoint(stage, evaluation, prediction, scores, {**clocks, 'preprocessing': state})
        tuned_frame, _ = read_fit(stage, evaluation)
    point_frame, _ = read_fit(point / 'outer_fits' / point_candidate, evaluation)
    outputs = {'hpo_same_radius_frozen': frozen_frame, 'hpo_spatial': tuned_frame,
               'hpo_selected': point_frame if selected['arm'] == POINT else tuned_frame}
    metadata = evaluation[METADATA].reset_index(drop=True).assign(feature_set=feature_set)
    print(f'{feature_set} outer={outer}: tuned {tuned["candidate"]}, selected {selected["arm"]}', flush=True)
    return (
        pd.concat([metadata.assign(arm=arm, prediction=frame.prediction) for arm, frame in outputs.items()],
                  ignore_index=True),
        pd.concat([pd.DataFrame([point_trial]), trials], ignore_index=True).assign(
            feature_set=feature_set, outer_fold=outer,
        ),
        {'feature_set': feature_set, 'outer_fold': outer, 'radius': radius, 'point_candidate': point_candidate,
         'tuned_candidate': tuned['candidate'], 'selected_arm': selected['arm'],
         'tuned_feasible': tuned['feasible'], 'selected_feasible': selected['feasible'],
         'tuned_safety_score': tuned['safety_score'], 'point_safety_score': point_trial['safety_score'],
         'same_radius_frozen_safety_score': screen['safety_score']},
    )


def track(output, predictions):
    client = mlflow.MlflowClient(tracking_uri=f'sqlite:///{output / "mlflow.db"}')
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = (experiment.experiment_id if experiment else
                     client.create_experiment(EXPERIMENT, artifact_location=(output / 'mlartifacts').as_uri()))
    runs = client.search_runs([experiment_id], filter_string=f"tags.protocol = '{PROTOCOL}'")
    if len(runs) > 1:
        raise ValueError('Duplicate MLflow runs for this nested spatial experiment')
    run = runs[0] if runs else client.create_run(
        experiment_id, tags={'mlflow.runName': output.name, 'protocol': PROTOCOL, 'scope': 'train_only_nested_oof'},
    )
    run_id = run.info.run_id
    if run.info.status != 'FINISHED':
        for key, value in {'seed': SEED, 'outer_folds': 3, 'inner_folds': 3, 'candidate_count': len(candidate_grid()),
                           'radius_options': ARMS, 'feature_sets': predictions.feature_set.unique().tolist()}.items():
            client.log_param(run_id, key, value)
        for (features, arm), frame in predictions.groupby(['feature_set', 'arm'], sort=True):
            metrics = raw_metrics(frame)
            for metric in ('critical_error_rate', 'critical_1_rate', 'critical_2_rate', 'false_alarm_rate',
                           'macro_f1', 'balanced_accuracy', 'release_rate'):
                client.log_metric(run_id, f'{features.replace("+", "__")}.{arm}.{metric}', metrics[metric])
        for name in ('plan.json', 'predictions.parquet', 'inner_trials.csv', 'selections.csv'):
            client.log_artifact(run_id, str(output / name))
        client.set_terminated(run_id)
    atomic_json(output / 'run.json', {'run_id': run_id, 'experiment_id': experiment_id, 'tracking': 'mlflow.db',
                                     'predictions_sha256': sha256_file(output / 'predictions.parquet')})


def run(nested, frozen, output, feature_sets, jobs, *, resume=False):
    nested, frozen, output = (Path(path).resolve() for path in (nested, frozen, output))
    if any(output.is_relative_to(source) for source in (nested, frozen)):
        raise ValueError('Write outside historical experiment directories')
    if jobs not in (1, 2) or len(set(feature_sets)) != len(feature_sets):
        raise ValueError('Choose one or two fit workers and unique feature sets')
    if resume and not (output / 'plan.json').is_file():
        raise FileNotFoundError('Resume requires an existing frozen HPO plan')
    output.mkdir(parents=True, exist_ok=resume)
    with (output / '.run.lock').open('a') as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'run.json').exists():
            raise FileExistsError('Experiment already finished')
        inputs = load_spatial_feature_inputs(nested)
        point_candidates = freeze_plan(output, nested, frozen, inputs, feature_sets)
        geometry = SpatialContext(inputs.train[['specimen_id', 'source_dataset', 'x', 'y']], radii=tuple(ARMS.values()))
        channels = inputs.features.frames['manual'].loc[:, list(SIGNAL_CHANNELS)]
        spatial = {arm: geometry.transform(channels, radii=(radius,), statistics=STATS) for arm, radius in ARMS.items()}
        frames, trials, choices = [], [], []
        for name in feature_sets:
            for fold in sorted(inputs.train.fold.unique()):
                frame, trial, choice = run_fold(output, nested, frozen, inputs, name, fold, spatial,
                                                point_candidates[f'{name}/{fold}'], jobs)
                frames.append(frame)
                trials.append(trial)
                choices.append(choice)
        predictions = pd.concat(frames, ignore_index=True)
        atomic_parquet(output / 'predictions.parquet', predictions)
        pd.concat(trials, ignore_index=True).to_csv(output / 'inner_trials.csv', index=False)
        pd.DataFrame(choices).to_csv(output / 'selections.csv', index=False)
        track(output, predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nested', type=Path, default=NESTED)
    parser.add_argument('--frozen', type=Path, default=FROZEN)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--feature-set', choices=FEATURES, action='append')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.nested, args.frozen, args.output_dir, args.feature_set or ['manual+tsfel'], args.jobs, resume=args.resume)


if __name__ == '__main__':
    main()
