"""Compare a fixed cardinal-neighbor ablation with the frozen radius experiment."""

import argparse
import fcntl
from importlib.metadata import version
from pathlib import Path
from platform import python_version

import mlflow
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.analysis.spatial_feature_report import summarize_spatial_features
from src.data.split_manifest import sha256_file
from src.experiments.spatial_feature_cv import (
    ARMS,
    FEATURES,
    METADATA,
    NESTED,
    POST,
    SEED,
    SIGNAL_CHANNELS,
    STATS,
    checkpoint_paths,
    fit_arm,
)
from src.experiments.spatial_feature_cv import OUTPUT as FIXED
from src.experiments.spatial_feature_inputs import (
    load_spatial_feature_inputs,
    load_spatial_feature_reference,
)
from src.experiments.svm_nested_hpo import atomic_json, atomic_parquet, read_fit
from src.features.spatial_context import SpatialContext
from src.features.specimen_inputs import read_json
from src.paths import EXPERIMENTS_ROOT, PROJECT_ROOT

FEATURE_SET = 'manual+tsfel'
ARM = 'axis_tight'
RADIUS = ARMS['tight']
OUTPUT = EXPERIMENTS_ROOT / 'specimen_spatial_features_axis_v1'


class CardinalContext(SpatialContext):
    """Retain the nearest measured point in each direction on the recorded raster axes."""

    def __init__(self, points):
        super().__init__(points, radii=(RADIUS,))
        graph = self.weights[RADIUS]
        rows = np.repeat(np.arange(len(points)), np.diff(graph.indptr))
        coordinates = points[['x', 'y']].to_numpy(dtype=float)
        dx, dy = (coordinates[graph.indices] - coordinates[rows]).T
        axial = (dx == 0) | (dy == 0)

        # Distinct measurements at the same axial distance and in the same
        # direction would be duplicate X/Y positions, rejected by the parent.
        distance = np.abs(dx[axial]) + np.abs(dy[axial])
        direction = 2 * (dx[axial] == 0).astype(int) + ((dx[axial] + dy[axial]) > 0)
        nearest = np.full((len(points), 4), np.inf)
        entries = (rows[axial], direction)
        np.minimum.at(nearest, entries, distance)
        keep = axial.copy()
        keep[axial] = distance == nearest[entries]
        graph.data[~keep] = 0
        graph.eliminate_zeros()


def _geometry_table(train, context):
    summaries = train[['specimen_id', 'source_dataset']].copy()
    summaries['neighbors'] = np.diff(context.weights[RADIUS].indptr)
    return (
        summaries.groupby(['source_dataset', 'specimen_id'], sort=True).neighbors
        .agg(points='size', min_neighbors='min', median_neighbors='median', max_neighbors='max',
             isolated_points=lambda counts: (counts == 0).sum())
        .reset_index()
    )


def _track(output, report):
    client = mlflow.MlflowClient(tracking_uri=f'sqlite:///{output / "mlflow.db"}')
    name = 'thz-spatial-svm-cardinal-ablation'
    stored = client.get_experiment_by_name(name)
    experiment = (
        stored.experiment_id if stored else client.create_experiment(
            name, artifact_location=(output / 'mlartifacts').as_uri(),
        )
    )
    run_id = client.create_run(experiment, tags={
        'mlflow.runName': output.name, 'scope': 'train_only_nested_oof',
    }).info.run_id
    try:
        for name, value in {'arm': ARM, 'feature_set': FEATURE_SET, 'seed': SEED, 'radius': RADIUS}.items():
            client.log_param(run_id, name, value)
        for row in report['pooled'].itertuples(index=False):
            for metric in ('critical_error_rate', 'cavity_miss_rate', 'uncertain_miss_rate', 'false_alarm_rate',
                           'macro_f1', 'balanced_accuracy', 'release_rate'):
                client.log_metric(run_id, f'{row.arm}.{metric}', getattr(row, metric))
        for name in ('plan.json', 'predictions.parquet', 'neighborhood_support.csv'):
            client.log_artifact(run_id, str(output / name))
        client.log_artifacts(run_id, str(output / 'report'), 'report')
    except Exception:
        client.set_terminated(run_id, 'FAILED')
        raise
    client.set_terminated(run_id)
    atomic_json(output / 'run.json', {'run_id': run_id, 'experiment_id': experiment, 'tracking': 'mlflow.db'})


def run(fixed, output, *, resume=False):
    fixed, output = Path(fixed).resolve(), Path(output).resolve()
    if any(output.is_relative_to(source) for source in (fixed, NESTED, POST)):
        raise ValueError('Write outside the existing experiment directories')
    if resume and not (output / 'plan.json').is_file():
        raise FileNotFoundError('Resume requires an existing frozen run plan')
    output.mkdir(parents=True, exist_ok=resume)
    with (output / '.run.lock').open('a') as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'run.json').exists():
            raise FileExistsError('Experiment already finished')
        return _run(fixed, output)


def _run(fixed, output):
    fixed_path = fixed / 'plan.json'
    fixed_plan = read_json(fixed_path)
    inputs = load_spatial_feature_inputs(NESTED)
    controls, hashes = checkpoint_paths(NESTED, inputs)
    if (fixed_plan['protocol'] != 'spatial_input_frozen_rbf_v1'
            or fixed_plan['inputs_sha256'] != {**inputs.hashes, **hashes}
            or any(sha256_file(PROJECT_ROOT / key) != sha for key, sha in fixed_plan['code_sha256'].items())):
        raise ValueError('Expected the same inputs and fit code as the frozen radius experiment')

    train = inputs.train
    folds = {int(fold): assessment for fold, assessment in train.groupby('fold', sort=True)}
    previous, ref_hashes = load_spatial_feature_reference(NESTED, POST, train, (FEATURE_SET,))
    reference_paths, reference_labels = {}, {}
    for fold, assessment in folds.items():
        path = fixed / 'fits' / FEATURE_SET / f'fold_{fold}' / 'outer' / 'tight'
        reference_labels[fold] = read_fit(path, assessment)[0].prediction.to_numpy()
        reference_paths.update({str(path / name): sha256_file(path / name)
                                for name in ('fit.json', 'predictions.parquet')})
    source_paths = (
        'src/experiments/spatial_feature_axis.py', 'src/experiments/spatial_inputs.py',
        'src/experiments/model_registry.py', 'src/features/specimen_inputs.py',
        'src/analysis/spatial_feature_report.py', 'src/analysis/spatial_postprocessing_report.py',
    )
    plan = {
        'protocol': 'spatial_input_cardinal_ablation_v1', 'feature_set': FEATURE_SET, 'arm': ARM,
        'seed': SEED, 'radius': RADIUS, 'weighting': 'same Gaussian weights as tight',
        'adjacency': 'nearest measured same-X or same-Y neighbor in each direction within tight radius',
        'statistics': list(STATS), 'signal_channels': list(SIGNAL_CHANNELS),
        'point_control': {str(fold): controls[FEATURE_SET, fold][1]['candidate'] for fold in folds},
        'inputs_sha256': {**inputs.hashes, **hashes, **ref_hashes, **reference_paths,
                          str(fixed_path): sha256_file(fixed_path)},
        'code_sha256': {**fixed_plan['code_sha256'],
                        **{key: sha256_file(PROJECT_ROOT / key) for key in source_paths}},
        'python': python_version(),
        'packages': {name: version(name) for name in fixed_plan['packages']},
    }
    plan_path = output / 'plan.json'
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError('Code or archived inputs changed since the experiment started')
    else:
        atomic_json(plan_path, plan)

    geometry = CardinalContext(train[['specimen_id', 'source_dataset', 'x', 'y']])
    channels = inputs.features.frames['manual'].loc[:, list(SIGNAL_CHANNELS)]
    spatial = geometry.transform(channels, statistics=STATS)
    _geometry_table(train, geometry).to_csv(output / 'neighborhood_support.csv', index=False)
    frames = [previous]
    for fold, assessment in folds.items():
        work, candidate = controls[FEATURE_SET, fold]
        point = read_fit(work / 'outer_fits' / candidate['candidate'], assessment)[0]
        prior = previous.loc[previous.fold.eq(fold) & previous.arm.eq('prior_native')]
        np.testing.assert_array_equal(point.prediction, prior.prediction)
        fit = train.loc[train.fold.ne(fold)]
        point_fit, point_eval, _ = inputs.features.prepare(fit, assessment, FEATURES[FEATURE_SET])
        directory = output / 'fits' / FEATURE_SET / f'fold_{fold}' / 'outer' / ARM
        evaluated = fit_arm(directory, fit, assessment, point_fit, point_eval, spatial, candidate, output)
        meta = assessment[METADATA].reset_index(drop=True).assign(feature_set=FEATURE_SET)
        frames.extend([meta.assign(arm=arm, prediction=prediction) for arm, prediction in {
            'point_only': point.prediction.to_numpy(),
            'tight': reference_labels[fold],
            ARM: evaluated.prediction.to_numpy(),
        }.items()])
        print(f'{FEATURE_SET} outer={fold}: {ARM} assessed', flush=True)
    predictions = pd.concat(frames, ignore_index=True)
    report = summarize_spatial_features(predictions)
    atomic_parquet(output / 'predictions.parquet', predictions)
    directory = output / 'report'
    directory.mkdir(exist_ok=True)
    for name, table in report.items():
        table.to_csv(directory / f'{name}.csv', index=False)
    _track(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed', type=Path, default=FIXED)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.fixed, args.output_dir, resume=args.resume)


if __name__ == '__main__':
    main()
