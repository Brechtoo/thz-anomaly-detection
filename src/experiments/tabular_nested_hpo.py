"""Tune five tabular model families using the archived SVM group assignments."""

import argparse
import fcntl
import json
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from platform import python_version

import mlflow
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from src.analysis.rbf_svm_oof_thresholds import frozen_train
from src.data.split_manifest import sha256_file
from src.evaluation.svm_nested_selection import (
    BASELINE_TOLERANCE,
    GUARDRAILS,
    apply_policy,
    calibrated_trials,
    choose_policy,
    fit_calibrator,
    predict_probabilities,
    raw_metrics,
)
from src.experiments.feature_grid import probabilities_in_label_order
from src.experiments.model_registry import create_model
from src.experiments.svm_nested_hpo import (
    FEATURES,
    KEYS,
    LABELS,
    SCORES,
    SEED,
    SOURCE,
    NestedFeatures,
    atomic_json,
    atomic_parquet,
    fit_checkpoint,
    prepare_fold,
    read_fit,
)
from src.experiments.svm_nested_hpo import OUTPUT as SVM
from src.experiments.tabular_nested_space import COSTS, MODELS, grid
from src.features.specimen_grid import TABLE_FILES
from src.features.specimen_inputs import (
    ANCHOR,
    EARLY,
    FOLDS,
    SPLITS,
    read_json,
    require_hash,
)
from src.paths import EXPERIMENTS_ROOT, FEATURE_TABLES_ROOT, PROJECT_ROOT

OUTPUT = EXPERIMENTS_ROOT / 'specimen_tabular_nested_hpo_v1'


def verified_assignments(train, path, digest):
    require_hash(path, digest)
    assignments = pd.read_parquet(path)
    if set(assignments.outer) != {0, 1, 2}:
        raise ValueError('Unexpected outer assignments')
    for outer, rows in assignments.groupby('outer'):
        expected = train.loc[train.fold.ne(outer), KEYS].sort_values('instance_id', ignore_index=True)
        observed = rows.drop(columns='fold').rename(columns={'original_fold': 'fold'})
        pd.testing.assert_frame_equal(observed[KEYS].sort_values('instance_id', ignore_index=True), expected)
        if set(rows.fold) != {0, 1, 2} or rows.groupby('specimen_id').fold.nunique().ne(1).any():
            raise ValueError(f'Invalid frozen inner split for outer fold {outer}')
        for inner in range(3):
            fit, held = prepare_fold(train, rows, inner)
            if set(fit.specimen_id) & set(held.specimen_id) or set(fit.label) != set(LABELS):
                raise ValueError(f'Invalid fit assignment for {outer}/{inner}')
            if any(set(held.loc[held.specimen_id.ne(body), 'label']) != set(LABELS)
                   for body in held.specimen_id.unique()):
                raise ValueError(f'Invalid calibration assignment for {outer}/{inner}')
    return assignments


def freeze(path, value):
    if path.exists():
        if read_json(path) != json.loads(json.dumps(value)):
            raise ValueError(f'Frozen plan or selection changed: {path}')
    else:
        atomic_json(path, value)


def freeze_plan(output, plan):
    """Preserve the November 24 plan while recording the float64 calibration revision."""
    path = output / 'plan.json'
    source = Path(__file__)
    original = 'd3c9313015420a74d3114057b0b90a60ec69ac41097b1ef2f65de7d715e0dc13'
    if path.exists() and read_json(path)['sha256'].get(str(source)) == original:
        require_hash(output / 'source/src/experiments/tabular_nested_hpo.py', original)
        freeze(path, plan | {'sha256': plan['sha256'] | {str(source): original}})
        freeze(output / 'calibration_precision_revision.json', {
            'parent_plan_sha256': sha256_file(path), 'source': str(source),
            'previous_sha256': original, 'sha256': plan['sha256'][str(source)],
            'change': 'Lossless conversion of checkpoint scores to float64 for calibration; no refits.',
        })
    else:
        freeze(path, plan)


def load_inputs(output, source, svm):
    svm_plan = read_json(svm / 'plan.json')
    if svm_plan['protocol'] != 'nested_train_only_v1' or svm_plan['features'] != {
        name: list(blocks) for name, blocks in FEATURES.items()
    }:
        raise ValueError('Expected the completed nested SVM protocol')
    for name, digest in svm_plan['source_sha256'].items():
        require_hash(PROJECT_ROOT / Path(name).relative_to(svm_plan['source_grid'].split('/results/')[0]), digest)
    train = frozen_train(source)
    assignments_path = svm / 'inner_assignments.parquet'
    assignments = verified_assignments(train, assignments_path, svm_plan['inner_assignment_sha256'])
    if len(train) != 34235 or train.specimen_id.nunique() != 23:
        raise ValueError('Unexpected training cohort size')
    paths = [svm / 'plan.json', assignments_path, source / 'report' / 'pooled_oof_metrics.csv',
             FOLDS / 'specimen_folds.csv', FOLDS / 'manifest.json', SPLITS / 'train.csv',
             ANCHOR / 'cohort_folds.csv', ANCHOR / 'features' / f'{EARLY}.npz',
             Path(__file__), Path(__file__).with_name('tabular_nested_space.py'),
             PROJECT_ROOT / 'src/analysis/tabular_nested_report.py']
    references = {}
    for combo in FEATURES:
        for fold in range(3):
            checkpoint = svm / 'outer' / combo / f'fold_{fold}'
            info, predictions = checkpoint / 'provenance.json', checkpoint / 'predictions.parquet'
            require_hash(predictions, read_json(info)['sha256'])
            reference = pd.read_parquet(predictions, columns=[*KEYS, 'prediction_regular', 'prediction_adapted'])
            held = train.loc[train.fold.eq(fold), KEYS].reset_index(drop=True)
            pd.testing.assert_frame_equal(reference[KEYS], held)
            references[combo, fold] = reference
            paths.extend((info, predictions))
    for block in ('tsfel', 'physical_v2'):
        path = FEATURE_TABLES_ROOT / TABLE_FILES[block]
        paths.extend((path, path.with_suffix('.json')))
    freeze_plan(output, {
        'protocol': 'tabular_native_and_gated_nested_train_v1', 'seed': SEED,
        'features': FEATURES, 'models': {model: grid(model) for model in MODELS},
        'guardrails': GUARDRAILS, 'baseline_tolerance': BASELINE_TOLERANCE,
        'reference': 'within-family registry defaults on the same feature set, HGB group-safe early_stopping=False',
        'python': python_version(), 'packages': {name: version(name) for name in
            ('numpy', 'pandas', 'scikit-learn', 'scipy', 'xgboost', 'lightgbm', 'pyarrow', 'mlflow')},
        'sha256': {str(path): sha256_file(path) for path in paths},
        'svm_source_sha256': svm_plan['source_sha256'],
    })
    return train, assignments, references


def fit_one(directory, model, candidate, fit, held, X, Z):
    if not directory.exists():
        estimator = create_model(model, seed=SEED)[-1].set_params(**candidate['params'])
        weights = COSTS[candidate['weights']]
        options = {'sample_weight': compute_sample_weight(weights, fit.label)} if weights else {}
        # LightGBM tracks supplied feature names in sklearn's estimator interface.
        if model == 'lightgbm':
            X, Z = pd.DataFrame(X), pd.DataFrame(Z)
        start = time.perf_counter()
        estimator.fit(X, fit.label.to_numpy(), **options)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        prediction = estimator.predict(Z)
        probabilities = probabilities_in_label_order(estimator.predict_proba(Z), estimator.classes_)
        scores = np.log(np.clip(probabilities, 1e-12, 1))
        fit_checkpoint(directory, held, prediction, scores, {
            'fit_seconds': fit_seconds, 'prediction_seconds': time.perf_counter() - start,
        })
    frame, _ = read_fit(directory, held)
    # XGBoost emits float32. Calibrate the saved values at the precision of the SVM protocol.
    return frame.astype(dict.fromkeys(SCORES, 'float64'), copy=False)


def policy_shortlist(raw):
    quality = raw.sort_values(['macro_f1', 'balanced_accuracy', 'critical_1_rate'], ascending=[False, False, True])
    best = quality.iloc[0]
    viable = raw.loc[raw.macro_f1.ge(best.macro_f1 - .08) & raw.balanced_accuracy.ge(best.balanced_accuracy - .08)]
    safe = viable.sort_values(['critical_1_rate', 'critical_error_rate', 'macro_f1'], ascending=[True, True, False])
    return list(dict.fromkeys([*quality.candidate.head(3), *safe.candidate.head(4), 'reference']))


def select(work, frames):
    pooled = {key: pd.concat(parts, ignore_index=True) for key, parts in frames.items()}
    raw = pd.DataFrame([{'candidate': key, **raw_metrics(frame)} for key, frame in pooled.items()])
    baseline = raw.loc[raw.candidate.eq('reference')].iloc[0].to_dict()
    raw.to_csv(work / 'inner_native.csv', index=False)
    policies = []
    for key in policy_shortlist(raw):
        path = work / f'policies-{key}.parquet'
        if not path.exists():
            atomic_parquet(path, calibrated_trials(pooled[key]))
        policies.append(pd.read_parquet(path).assign(candidate=key))
    choices = {
        'regular': choose_policy(raw, baseline),
        'adapted': choose_policy(pd.concat([raw, *policies], ignore_index=True), baseline),
    }
    freeze(work / 'selection.json', choices)
    return choices, pooled


def outer_fold(output, model, combo, outer, train, assignments, features, pool):
    work = output / 'fits' / model / combo / f'outer_{outer}'
    work.mkdir(parents=True, exist_ok=True)
    held = train.loc[train.fold.eq(outer)]
    path = work / 'outer_predictions.parquet'
    info = work / 'outer_summary.json'
    if info.exists():
        require_hash(path, read_json(info)['sha256'])
        result = pd.read_parquet(path)
        pd.testing.assert_frame_equal(result[KEYS], held[KEYS].reset_index(drop=True))
        return result

    candidates = {row['candidate']: row for row in grid(model)}
    frames = {key: [] for key in candidates}
    for inner in range(3):
        fit, evaluation = prepare_fold(train, assignments, inner)
        X, Z, state = features.prepare(fit, evaluation, FEATURES[combo])
        freeze(work / f'preprocess-inner-{inner}.json', state)
        jobs = {key: pool.submit(fit_one, work / f'inner_{inner}' / key, model, candidate,
                                 fit, evaluation, X, Z) for key, candidate in candidates.items()}
        for key, job in jobs.items():
            frames[key].append(job.result())
        print(f'{model} {combo} outer={outer} inner={inner}: {len(jobs)} verified fits', flush=True)
    choices, pooled = select(work, frames)
    first = train.loc[train.fold.ne(outer)]
    X, Z, state = features.prepare(first, held, FEATURES[combo])
    freeze(work / 'preprocess-outer.json', state)
    fits = {
        key: fit_one(work / 'outer' / key, model, candidates[key], first, held, X, Z)
        for key in dict.fromkeys(choice['candidate'] for choice in choices.values())
    }
    result = held[KEYS].reset_index(drop=True).copy()
    result['prediction_regular'] = fits[choices['regular']['candidate']].prediction
    selected = choices['adapted']
    key = selected['candidate']
    calibrator = fit_calibrator(pooled[key][SCORES].to_numpy(), pooled[key].label.to_numpy(),
                                method=selected['calibration'])
    result['prediction_hard'] = fits[key].prediction
    proba = predict_probabilities(calibrator, fits[key][SCORES].to_numpy())
    result['prediction_adapted'] = apply_policy(fits[key].prediction.to_numpy(), proba, selected)
    atomic_parquet(path, result)
    atomic_json(info, {'sha256': sha256_file(path)})
    return result


def run(output, source, svm, jobs):
    from src.analysis.tabular_nested_report import write_report

    if any(output.is_relative_to(path) for path in (source, svm)):
        raise ValueError('Choose a separate output directory')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'completed.json').exists():
            raise FileExistsError('Experiment already completed')
        train, assignments, references = load_inputs(output, source, svm)
        features = NestedFeatures(train)
        mlflow.set_tracking_uri(f'sqlite:///{output / "mlflow.db"}')
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name('thz-tabular-nested-hpo')
        if not experiment:
            experiment_id = client.create_experiment(
                'thz-tabular-nested-hpo', artifact_location=(output / 'mlartifacts').as_uri(),
            )
            experiment = client.get_experiment(experiment_id)
        run_path = output / 'run.json'
        run_id = read_json(run_path)['run_id'] if run_path.exists() else None
        with mlflow.start_run(run_id=run_id, experiment_id=experiment.experiment_id, run_name=output.name) as run:
            freeze(run_path, {'run_id': run.info.run_id, 'experiment_id': experiment.experiment_id})
            mlflow.set_tag('scope', 'frozen_train_only_nested_oof')
            mlflow.log_params({'seed': SEED, 'families': ' '.join(MODELS), 'inner_folds': 3, 'outer_folds': 3,
                               'feature_sets': len(FEATURES), 'candidates_per_family_and_set': 16})
            mlflow.log_artifact(str(output / 'plan.json'))
            evaluated = {}
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                for model in MODELS:
                    for combo in FEATURES:
                        for outer in range(3):
                            inner = assignments.loc[assignments.outer.eq(outer)]
                            evaluated[model, combo, outer] = outer_fold(
                                output, model, combo, outer, train, inner, features, pool,
                            )
                            mlflow.log_metric('completed_outer_assessments', len(evaluated), step=len(evaluated))
                            print(f'{model} {combo} outer={outer}: assessed', flush=True)
                        write_report(output, evaluated, references, completed=False)
            report = write_report(output, evaluated, references, completed=True)
            mlflow.log_artifacts(str(output / 'report'), 'report')
            for row in report.loc[report.scope.eq('pooled')].itertuples(index=False):
                prefix = f'{row.model}.{row.features.replace("+", "_")}.{row.mode}'
                for key in ('critical_error_rate', 'macro_f1', 'balanced_accuracy', 'false_alarm_rate'):
                    mlflow.log_metric(f'{prefix}.{key}', getattr(row, key))
        atomic_json(output / 'completed.json', {'run_id': run.info.run_id})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--svm', type=Path, default=SVM)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3, 4), default=3)
    args = parser.parse_args()
    run(args.output.resolve(), args.source.resolve(), args.svm.resolve(), args.jobs)


if __name__ == '__main__':
    main()
