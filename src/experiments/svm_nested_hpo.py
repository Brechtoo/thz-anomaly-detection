"""Nested, training-only RBF-SVM tuning on the frozen HeRAI specimen folds."""

import argparse
import fcntl
import itertools
import json
import time
import traceback
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from src.analysis.rbf_svm_oof_thresholds import frozen_train
from src.data.split_manifest import sha256_file
from src.evaluation.metrics import ERROR_METRICS_VERSION
from src.experiments.feature_grid import KEYS, LABELS
from src.experiments.model_registry import create_model
from src.experiments.svm_nested_kernel import iter_rbf_fits
from src.experiments.tracking import set_project_experiment
from src.features.specimen_grid import EXPECTED_WIDTH, TABLE_FILES, GridFeatures
from src.features.specimen_inputs import (
    ANCHOR,
    FOLDS,
    SPLITS,
    load_early_manual_train,
    read_json,
    require_hash,
    write_json,
)
from src.paths import EXPERIMENTS_ROOT, FEATURE_TABLES_ROOT, PROJECT_ROOT

SEED = 42
EXPERIMENT = 'thz-specimen-rbf-svm-nested-hpo'
FEATURES = {
    'manual+tsfel': ('manual', 'tsfel'),
    'manual+tsfel+physical_v2': ('manual', 'tsfel', 'physical_v2'),
    'manual': ('manual',),
}
SOURCE = EXPERIMENTS_ROOT / 'specimen_feature_grid_v1'
OUTPUT = EXPERIMENTS_ROOT / 'specimen_svm_nested_hpo_v1'
SCORES = [f'score_{label}' for label in LABELS]
WEIGHTS = {'balanced': 'balanced', 'uniform': None, 'cavity_cost': {0: 1., 1: 2.5, 2: 1.6, 3: 0.85}}
REFERENCE = 'c0.3_g0.001_balanced'


def candidate_grid():
    center = itertools.product((0.1, 0.3, 1.), (0.0003, 0.001, 0.003), WEIGHTS)
    probes = ((c, g, 'balanced') for c, g in [
        (0.03, 0.001), (3., 0.001), (10., 0.001), (0.3, 0.0001), (0.3, 0.01),
    ])
    candidates = [
        {'candidate': f'c{c:g}_g{g:g}_{weights}', 'C': c, 'gamma': g,
         'weights': weights, 'class_weight': WEIGHTS[weights]}
        for c, g, weights in itertools.chain(center, probes)
    ]
    return sorted(candidates, key=lambda item: (item['candidate'] != REFERENCE, item['gamma'], item['C'], item['weights']))


def inner_assignments(train):
    """Assign inner blocks using outer-training IDs, labels and specimen groups only."""
    parts = []
    for outer in sorted(train.fold.unique()):
        allowed = train.loc[train.fold.ne(outer)]
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + int(outer))
        assignments = allowed[KEYS].rename(columns={'fold': 'original_fold'}).assign(outer=int(outer), fold=-1)
        for inner, (in_fit, in_eval) in enumerate(splitter.split(allowed, allowed.label, allowed.specimen_id)):
            fit, evaluation = allowed.iloc[in_fit], allowed.iloc[in_eval]
            if (set(fit.specimen_id) & set(evaluation.specimen_id) or set(fit.label) != set(LABELS)
                    or set(evaluation.label) != set(LABELS)):
                raise ValueError(f'Invalid inner partition {outer}/{inner}')
            for specimen in evaluation.specimen_id.unique():
                if set(evaluation.loc[evaluation.specimen_id.ne(specimen), 'label']) != set(LABELS):
                    raise ValueError(f'Insufficient calibration labels without {specimen} in {outer}/{inner}')
            assignments.loc[evaluation.index, 'fold'] = inner
        if assignments.fold.eq(-1).any() or len(assignments) != assignments.instance_id.nunique():
            raise ValueError(f'Inner fold coverage failed for {outer}')
        parts.append(assignments)
    return pd.concat(parts, ignore_index=True)


def load_training_block(name, train):
    """Use only train rows of the verified, signal-wise feature tables."""
    path = FEATURE_TABLES_ROOT / TABLE_FILES[name]
    manifest = read_json(path.with_suffix('.json'))
    require_hash(path, manifest['parquet_sha256'])
    columns = manifest['retained_features']
    if len(columns) != EXPECTED_WIDTH[name] or len(set(columns)) != len(columns):
        raise ValueError(f'Unexpected feature width or duplicate columns in {path}')
    metadata = ['instance_id', 'specimen_id', 'source_dataset', 'label', 'split', 'cv_fold']
    table = pd.read_parquet(path, filters=[('split', '==', 'train')]).sort_values('instance_id', ignore_index=True)
    if set(table.columns) != set(metadata) | set(columns):
        raise ValueError(f'Unexpected training feature schema in {path}')
    labels = table[metadata].rename(columns={'cv_fold': 'fold'})
    pd.testing.assert_frame_equal(labels, train[labels.columns], check_dtype=False, check_exact=True)
    if not all(pd.api.types.is_numeric_dtype(table[col]) for col in columns):
        raise ValueError(f'Non-numeric training features in {path}')
    return table[columns]


class NestedFeatures(GridFeatures):
    """Reuse the Grid block preparation with explicit fit and assessment bodies."""

    def __init__(self, train):
        self.train = train
        self.frames = {'manual': load_early_manual_train(train)}
        if self.frames['manual'].shape != (len(train), EXPECTED_WIDTH['manual']):
            raise ValueError('Incorrect early manual training matrix')
        for name in ('tsfel', 'physical_v2'):
            self.frames[name] = load_training_block(name, train)
        self._blocks = lru_cache(maxsize=6)(self._prepare_block)

    def _cohort(self, partition):
        fit, evaluation = partition
        cohort = self.train.loc[self.train.specimen_id.isin(fit + evaluation)]
        return cohort, cohort.specimen_id.isin(fit)

    def prepare(self, fit, evaluation, names):
        fit_bodies = tuple(sorted(fit.specimen_id.unique()))
        eval_bodies = tuple(sorted(evaluation.specimen_id.unique()))
        if set(fit_bodies) & set(eval_bodies):
            raise ValueError('Specimen overlap between preprocessing fit and assessment')
        X, state, cohort, fit_mask = self._assemble((fit_bodies, eval_bodies), names)
        pd.testing.assert_index_equal(cohort.index[fit_mask], fit.index)
        pd.testing.assert_index_equal(cohort.index[~fit_mask], evaluation.index)
        scaler = create_model('rbf_svm', seed=SEED)[:-1]
        X_fit = np.ascontiguousarray(scaler.fit_transform(X[fit_mask]))
        X_eval = np.ascontiguousarray(scaler.transform(X[~fit_mask]))
        state.update(fit_specimens=list(fit_bodies), assessment_specimens=list(eval_bodies),
                     scaler_mean=scaler['scale'].mean_.tolist(), scaler_scale=scaler['scale'].scale_.tolist(),
                     variance_support=scaler['constants'].get_support().tolist())
        return X_fit, X_eval, state


def atomic_json(path, value):
    with TemporaryDirectory(dir=path.parent, prefix='.writing-') as temp:
        staged = Path(temp) / path.name
        write_json(staged, value)
        staged.replace(path)


def atomic_parquet(path, frame):
    with TemporaryDirectory(dir=path.parent, prefix='.writing-') as temp:
        staged = Path(temp) / path.name
        frame.to_parquet(staged, index=False, compression='zstd')
        staged.replace(path)


def freeze_plan(output, source, assignments, jobs):
    source_paths = [
        Path(__file__), PROJECT_ROOT / 'src/experiments/svm_nested_kernel.py',
        PROJECT_ROOT / 'src/evaluation/svm_nested_selection.py',
        PROJECT_ROOT / 'src/features/specimen_grid.py', PROJECT_ROOT / 'src/features/specimen_inputs.py',
        PROJECT_ROOT / 'src/experiments/model_registry.py', PROJECT_ROOT / 'src/evaluation/metrics.py',
        PROJECT_ROOT / 'src/analysis/rbf_svm_oof_thresholds.py',
        ANCHOR / 'extraction_manifest.json', ANCHOR / 'plan.json',
        FOLDS / 'manifest.json', SPLITS / 'train.csv',
        source / 'plan.json', source / 'cohort.parquet',
        *(FEATURE_TABLES_ROOT / TABLE_FILES[name].replace('.parquet', '.json') for name in ('tsfel', 'physical_v2')),
    ]
    folds_path = output / 'inner_assignments.parquet'
    if folds_path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(folds_path), assignments)
    else:
        atomic_parquet(folds_path, assignments)
    plan = {
        'protocol': 'nested_train_only_v1', 'source_grid': str(source), 'seed': SEED, 'kernel_jobs': jobs,
        'class_order': LABELS.tolist(), 'error_metrics': ERROR_METRICS_VERSION,
        'features': {key: list(value) for key, value in FEATURES.items()},
        'inner_assignment_sha256': sha256_file(folds_path), 'inner_seed': '42 + outer fold',
        'candidates': candidate_grid(),
        'calibration': 'same-inner-block leave-assessed-specimen-out; final fit on inner OOF',
        'shortlist': 'union of 3 regular, up to 4 safety from quality-constrained hard predictions, and fixed reference',
        'source_sha256': {str(p): sha256_file(p) for p in source_paths},
        'packages': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'scipy', 'pyarrow', 'mlflow')},
    }
    # Normalize weighting dictionaries to their JSON representation before checking on resume.
    path = output / 'plan.json'
    if path.exists():
        if read_json(path) != json.loads(json.dumps(plan)):
            raise ValueError('Frozen HPO plan/input/code differs from the checkpoint')
    else:
        atomic_json(path, plan)


def fit_checkpoint(directory, evaluation, prediction, scores, timings):
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent, prefix='.fitting-') as temp:
        staged = Path(temp) / 'checkpoint'
        staged.mkdir()
        frame = evaluation[KEYS].reset_index(drop=True).assign(prediction=prediction)
        frame[SCORES] = scores
        frame.to_parquet(staged / 'predictions.parquet', index=False, compression='zstd')
        write_json(staged / 'fit.json', {**timings, 'sha256': sha256_file(staged / 'predictions.parquet')})
        staged.rename(directory)


def read_fit(directory, expected):
    path = directory / 'predictions.parquet'
    info = read_json(directory / 'fit.json')
    require_hash(path, info['sha256'])
    frame = pd.read_parquet(path)
    pd.testing.assert_frame_equal(frame[KEYS], expected[KEYS].reset_index(drop=True))
    if not np.isfinite(frame[SCORES]).all(axis=None) or not frame.prediction.isin(LABELS).all():
        raise ValueError(f'Invalid SVM predictions in {directory}')
    return frame, info


def prepare_fold(train, train_inner, inner):
    by_id = train_inner.set_index('instance_id')['fold']
    usable = train.loc[train.instance_id.isin(by_id.index)]
    group = usable.instance_id.map(by_id)
    fit, evaluation = usable.loc[group.ne(inner)], usable.loc[group.eq(inner)]
    return fit, evaluation.assign(fold=inner)


def fit_inner(output, work, train, train_inner, features, combo, outer, candidates, jobs):
    for inner in range(3):
        directory = work / f'inner_{inner}'
        directory.mkdir(parents=True, exist_ok=True)
        remaining = [candidate for candidate in candidates if not (directory / candidate['candidate']).exists()]
        if not remaining:
            continue
        fit, evaluation = prepare_fold(train, train_inner, inner)
        X_fit, X_eval, state = features.prepare(fit, evaluation, FEATURES[combo])
        atomic_json(directory / 'preprocessing.json', state)
        print(f'{combo} outer={outer} inner={inner} {len(fit)}→{len(evaluation)}, {len(remaining)} fits', flush=True)

        def on_error(candidate, error):
            failure = directory / 'failures'
            failure.mkdir(exist_ok=True)
            atomic_json(failure / f"{candidate['candidate']}.json", {'error': repr(error)})
            print(f"  {candidate['candidate']} failed: {error}", flush=True)

        for candidate, pred, scores, clocks in iter_rbf_fits(
            X_fit, fit.label.to_numpy(), X_eval, remaining, output, seed=SEED, jobs=jobs, on_error=on_error,
        ):
            fit_checkpoint(directory / candidate['candidate'], evaluation, pred, scores, clocks)
            print(f"  {candidate['candidate']} {clocks['training_seconds']:.1f}s", flush=True)


def pooled_inner(work, assignments, train, candidate):
    frames, clocks = [], []
    for inner in range(3):
        _, evaluation = prepare_fold(train, assignments, inner)
        frame, info = read_fit(work / f'inner_{inner}' / candidate, evaluation)
        frames.append(frame)
        clocks.append(info)
    return pd.concat(frames, ignore_index=True), clocks


def unique_log(client, parent, log_key, name, params, metrics, artifacts, tags=None):
    """Restore a previously interrupted child without duplicating completed children."""
    runs = client.search_runs([parent.info.experiment_id],
                              filter_string=f"tags.batch = '{parent.info.run_id}' AND tags.log_key = '{log_key}'")
    if len(runs) > 1:
        raise ValueError(f'Duplicate MLflow checkpoint key: {log_key}')
    if runs and runs[0].info.status == 'FINISHED':
        return runs[0].info.run_id
    with mlflow.start_run(run_id=runs[0].info.run_id if runs else None, run_name=name, nested=True) as run:
        mlflow.set_tags({'mlflow.parentRunId': parent.info.run_id, 'batch': parent.info.run_id,
                         'log_key': log_key, **(tags or {})})
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items() if np.isfinite(value)})
        for path in artifacts:
            if path.is_dir():
                mlflow.log_artifacts(str(path), path.name)
            else:
                mlflow.log_artifact(str(path))
        return run.info.run_id


def shortlist(raw):
    # Include both high-quality four-class classifiers and candidates with useful raw safety.
    ranked = raw.sort_values(['macro_f1', 'balanced_accuracy', 'critical_1_rate', 'candidate'],
                             ascending=[False, False, True, True])
    best = ranked.iloc[0]
    viable = raw.loc[raw.macro_f1.ge(best.macro_f1 - 0.08) & raw.balanced_accuracy.ge(best.balanced_accuracy - 0.08)]
    safe = viable.sort_values(['critical_1_rate', 'critical_error_rate', 'macro_f1', 'candidate'],
                              ascending=[True, True, False, True])
    return list(dict.fromkeys([*ranked.candidate.head(3), *safe.candidate.head(4), REFERENCE]))


def choose_inner(work, assignments, train, combo, outer, candidates, client, parent):
    from src.evaluation.svm_nested_selection import calibrated_trials, choose_policy, raw_metrics

    candidates = [candidate for candidate in candidates
                  if all((work / f'inner_{fold}' / candidate['candidate']).exists() for fold in range(3))]
    if not any(candidate['candidate'] == REFERENCE for candidate in candidates):
        raise ValueError('Frozen reference lacked at least one inner fold')
    raw_rows = []
    for candidate in candidates:
        key = candidate['candidate']
        frame, clocks = pooled_inner(work, assignments, train, key)
        raw = raw_metrics(frame)
        trial = work / 'candidates' / key
        trial.mkdir(parents=True, exist_ok=True)
        atomic_json(trial / 'raw.json', raw)
        atomic_json(trial / 'timings.json', clocks)
        raw_rows.append({**raw, 'candidate': key})
    raw = pd.DataFrame(raw_rows)
    reference = raw.loc[raw.candidate.eq(REFERENCE)].iloc[0].to_dict()
    names = shortlist(raw)
    calibrated = []
    for key in names:
        trial = work / 'candidates' / key
        path = trial / 'policies.parquet'
        if not path.exists():
            print(f'{combo} outer={outer} calibration {key}', flush=True)
            frame, _ = pooled_inner(work, assignments, train, key)
            atomic_parquet(path, calibrated_trials(frame))
        calibrated.append(pd.read_parquet(path).assign(candidate=key))
    policies = pd.concat(calibrated, ignore_index=True)
    for candidate in candidates:
        key = candidate['candidate']
        trial = work / 'candidates' / key
        values = raw.loc[raw.candidate.eq(key)].iloc[0]
        metrics = {name: val for name, val in values.items() if isinstance(val, (int, float, np.number, bool))}
        log_params = {**candidate, 'features': combo, 'outer_fold': outer, 'seed': SEED,
                      'used_in_calibration_shortlist': key in names}
        unique_log(client, parent, f'{combo}-{outer}-inner-{key}', f'{combo}/outer_{outer}/inner/{key}',
                   log_params, metrics, [trial], {'stage': 'inner_search', 'features': combo})
    regular = choose_policy(raw, reference)
    adapted = choose_policy(pd.concat([raw, policies], ignore_index=True), reference)
    selected = {'regular': regular, 'adapted': adapted}
    rows = {key: pooled_inner(work, assignments, train, key)[0]
            for key in {row['candidate'] for row in selected.values()}}
    return selected, rows


def finish_outer(output, work, first, held, features, combo, outer, configs, selections, rows, jobs, client, parent):
    from src.evaluation.svm_nested_selection import (
        apply_policy,
        decision_metrics,
        fit_calibrator,
        predict_probabilities,
    )

    checkpoint = output / 'outer' / combo / f'fold_{outer}'
    if not checkpoint.exists():
        work.mkdir(parents=True, exist_ok=True)
        # Selections are persisted before model fitting or prediction on the outer fold.
        selection_path = work / 'selection.json'
        if selection_path.exists():
            if read_json(selection_path) != selections:
                raise ValueError('Inner selection differs from the persisted pre-outer choice')
        else:
            atomic_json(selection_path, selections)
        wanted = {row['candidate'] for row in selections.values()}
        chosen = [candidate for candidate in configs if candidate['candidate'] in wanted]
        remaining = [candidate for candidate in chosen if not (
            work / 'outer_fits' / candidate['candidate']).exists()]
        if remaining:
            X_fit, X_eval, state = features.prepare(first, held, FEATURES[combo])
            atomic_json(work / 'outer_preprocessing.json', state)
            for candidate, pred, scores, clocks in iter_rbf_fits(
                X_fit, first.label.to_numpy(), X_eval, remaining, output, seed=SEED, jobs=jobs,
            ):
                fit_checkpoint(work / 'outer_fits' / candidate['candidate'], held, pred, scores, clocks)
        frame = held[KEYS].reset_index(drop=True).copy()
        outer_metrics, selection_rows, calib_details = {}, [], {}
        for kind, selected in selections.items():
            key = selected['candidate']
            metadata = next(candidate for candidate in chosen if candidate['candidate'] == key)
            predictions, timings = read_fit(work / 'outer_fits' / key, held)
            scores = predictions[SCORES].to_numpy()
            hard = predictions.prediction.to_numpy()
            inner_rows = rows[key]
            calibrator = fit_calibrator(inner_rows[SCORES].to_numpy(), inner_rows.label.to_numpy(),
                                        method=selected['calibration'])
            probabilities = predict_probabilities(calibrator, scores)
            prefix = 'regular_' if kind == 'regular' else ''
            frame[[f'{prefix}p_{k}' for k in LABELS]] = probabilities
            frame[[f'{prefix}score_{k}' for k in LABELS]] = scores
            if kind == 'regular':
                frame['prediction_regular'] = hard
            else:
                frame['prediction_hard'] = hard
                frame['prediction_adapted'] = apply_policy(hard, probabilities, selected)
                frame['prediction_review'] = apply_policy(hard, probabilities, {**selected, 'fallback': 'review'})
            selection_rows.append({'features': combo, 'fold': outer, 'kind': kind, **metadata,
                                   **{name: selected[name] for name in ('gate', 'threshold', 'fallback', 'calibration')},
                                   **{f'inner_{name}': value for name, value in selected.items()}})
            calib_details[kind] = {'method': selected['calibration'], 'inputs': len(inner_rows), 'fit_times': timings}
        for mode in ('regular', 'hard', 'adapted'):
            for name, val in decision_metrics(frame.label.to_numpy(), frame[f'prediction_{mode}'].to_numpy()).items():
                outer_metrics[f'{mode}.{name}'] = val
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=checkpoint.parent, prefix='.outer-') as temp:
            staged = Path(temp) / 'fold'
            staged.mkdir()
            frame.to_parquet(staged / 'predictions.parquet', index=False, compression='zstd')
            pd.DataFrame(selection_rows).to_csv(staged / 'selections.csv', index=False)
            write_json(staged / 'metrics.json', outer_metrics)
            write_json(staged / 'provenance.json', {'sha256': sha256_file(staged / 'predictions.parquet'),
                                                   'calibration': calib_details})
            staged.rename(checkpoint)
    require_hash(checkpoint / 'predictions.parquet', read_json(checkpoint / 'provenance.json')['sha256'])
    unique_log(client, parent, f'{combo}-{outer}-outer', f'{combo}/outer_{outer}/evaluation',
               {'features': combo, 'outer_fold': outer, 'seed': SEED, 'scope': 'outer_train_oof'},
               read_json(checkpoint / 'metrics.json'), [checkpoint, work / 'outer_preprocessing.json'],
               {'stage': 'outer_evaluation', 'features': combo})


def update_aggregate(output):
    result = output / 'outer_oof'
    result.mkdir(exist_ok=True)
    selections = []
    for combo in FEATURES:
        pieces = sorted((output / 'outer' / combo).glob('fold_*'))
        if pieces:
            for path in pieces:
                require_hash(path / 'predictions.parquet', read_json(path / 'provenance.json')['sha256'])
            frame = pd.concat([pd.read_parquet(part / 'predictions.parquet') for part in pieces], ignore_index=True)
            atomic_parquet(result / f'{combo}.parquet', frame.sort_values('instance_id', ignore_index=True))
            selections.extend(pd.read_csv(part / 'selections.csv') for part in pieces)
    if selections:
        with TemporaryDirectory(dir=output, prefix='.aggregate-') as temp:
            path = Path(temp) / 'selections.csv'
            pd.concat(selections, ignore_index=True).to_csv(path, index=False)
            path.replace(output / path.name)


def run(source, output, jobs):
    if output.is_relative_to(source):
        raise ValueError('Output must be separate from the historical Grid')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.runner.lock').open('a') as lock, threadpool_limits(1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train = frozen_train(source)
        assignments = inner_assignments(train)
        features = NestedFeatures(train)
        configs = candidate_grid()
        freeze_plan(output, source, assignments, jobs)
        experiment = set_project_experiment(EXPERIMENT)
        client = mlflow.MlflowClient()
        mlflow.sklearn.autolog(disable=True)
        info = output / 'run.json'
        previous = read_json(info) if info.exists() else {}
        if previous and previous['experiment_id'] != experiment.experiment_id:
            raise ValueError('Unexpected MLflow experiment on resume')
        with mlflow.start_run(run_id=previous.get('run_id'), run_name=output.name) as parent:
            if not previous:
                atomic_json(info, {'run_id': parent.info.run_id, 'experiment_id': experiment.experiment_id})
            mlflow.set_tags({'scope': 'nested_training_oof', 'stage': 'batch', 'holdouts': 'locked'})
            mlflow.log_params({'seed': SEED, 'candidates': len(configs), 'inner_folds': 3, 'outer_folds': 3,
                               'error_metrics': ERROR_METRICS_VERSION, 'kernel_jobs': jobs})
            mlflow.log_artifact(str(output / 'plan.json'))
            mlflow.log_artifact(str(output / 'inner_assignments.parquet'))
            failures = []
            from src.analysis.svm_nested_report import write_report

            for combo in FEATURES:
                for outer in range(3):
                    work = output / 'search' / combo / f'outer_{outer}'
                    work.mkdir(parents=True, exist_ok=True)
                    try:
                        started = time.perf_counter()
                        first, held = train.loc[train.fold.ne(outer)], train.loc[train.fold.eq(outer)]
                        training = assignments.loc[assignments.outer.eq(outer)]
                        if (output / 'outer' / combo / f'fold_{outer}').exists():
                            selections, rows = {}, {}
                        else:
                            fit_inner(output, work, train, training, features, combo, outer, configs, jobs)
                            selections, rows = choose_inner(work, training, train, combo, outer,
                                                            configs, client, parent)
                        finish_outer(output, work, first, held, features, combo, outer, configs,
                                     selections, rows, jobs, client, parent)
                        update_aggregate(output)
                        write_report(output, source, finalized=False)
                        print(f'{combo} outer={outer} saved, session {time.perf_counter() - started:.1f}s', flush=True)
                    except Exception:
                        atomic_json(work / 'failure.json', {'traceback': traceback.format_exc()})
                        print(traceback.format_exc(), flush=True)
                        failures.append((combo, outer))
            write_report(output, source, finalized=not failures)
            mlflow.log_artifacts(str(output / 'report'), 'report')
            mlflow.log_metric('complete_outer_folds', 9 - len(failures))
            mlflow.log_metric('completed_inner_fits', len(list(output.glob('search/*/*/inner_*/c*/fit.json'))))
            if failures:
                raise RuntimeError(f'Incomplete outer evaluations: {failures}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    run(args.source.resolve(), args.output_dir.resolve(), args.jobs)


if __name__ == '__main__':
    main()
