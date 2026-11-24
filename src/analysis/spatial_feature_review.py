"""Evaluate completed spatial SVM runs without retraining any model."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.analysis.spatial_feature_report import (
    KEYS,
    POINT_ONLY,
    REQUIRED,
    summarize_spatial_features,
)
from src.analysis.spatial_postprocessing_report import (
    IDENTITY,
    _bootstrap,
    _confusion,
    _diagnostic_slices,
)
from src.data.split_manifest import sha256_file
from src.features.specimen_inputs import read_json, write_json

FIXED_ARMS = {POINT_ONLY, 'tight', 'wide', 'selected', 'prior_native', 'prior_rescue'}
HPO_ARMS = {'hpo_spatial', 'hpo_same_radius_frozen', 'hpo_selected'}
POINT_HPO = 'hpo_point_matched'
AXIS_ARM = 'axis_tight'
AXIS_ARMS = {AXIS_ARM, POINT_ONLY, 'tight', 'prior_native', 'prior_rescue'}
MATCHED_PAIRS = (
    ('hpo_spatial', 'hpo_same_radius_frozen'),
    ('hpo_spatial', POINT_HPO),
    ('hpo_selected', 'selected'),
    ('hpo_selected', POINT_HPO),
    (AXIS_ARM, 'tight'),
)
CASE_BODIES = ('messreihe_1_pk_09', 'messreihe_3_pk_06')


def _group_slices(frame):
    yield {'scope': 'pooled'}, slice(None)
    for fold, positions in frame.groupby('fold', sort=True).indices.items():
        yield {'scope': 'fold', 'fold': fold}, positions
    for (source, specimen), positions in frame.groupby(IDENTITY, sort=True).indices.items():
        yield {'scope': 'specimen', 'fold': frame.fold.iloc[positions[0]],
               'source_dataset': source, 'specimen_id': specimen}, positions


def _error_changes(truth, prediction, reference, slices, names):
    risk = np.isin(truth, [1, 2])
    clear = truth == 3
    released, reference_released = prediction == 3, reference == 3
    false_alarm = np.isin(prediction, [1, 2])
    reference_false_alarm = np.isin(reference, [1, 2])
    errors = (
        ('critical_1_or_2_to_3', risk, released, reference_released),
        ('cavity_1_to_3', truth == 1, released, reference_released),
        ('uncertain_2_to_3', truth == 2, released, reference_released),
        ('false_alarm_3_to_1_or_2', clear, false_alarm, reference_false_alarm),
        ('poor_measurement_0_to_3', truth == 0, released, reference_released),
    )
    for name, support, predicted_error, reference_error in errors:
        transitions = np.column_stack((
            support,
            support & reference_error & predicted_error,
            support & reference_error & ~predicted_error,
            support & ~reference_error & predicted_error,
        ))
        for scope, indices in slices:
            n, retained, prevented, introduced = transitions[indices].sum(axis=0)
            if n:
                yield {
                    **names, **scope, 'error': name, 'support': n, 'retained': retained,
                    'prevented': prevented, 'introduced': introduced,
                }


def review_predictions(predictions, *, draws=2000, seed=42):
    """Compare arms on the same outer decisions and audit where errors move."""
    tables = summarize_spatial_features(predictions, draws=draws, seed=seed)
    confusion, changes, matched_intervals = [], [], []
    for feature_set, rows in predictions.groupby('feature_set', sort=True):
        arms = {arm: group.sort_values('instance_id', ignore_index=True)
                for arm, group in rows.groupby('arm', sort=True)}
        control = arms[POINT_ONLY]
        truth = control.label.to_numpy()
        groups = tuple(_group_slices(control))
        diagnostics = tuple(
            ({'scope': 'diagnostic', 'slice': name}, mask)
            for name, mask in _diagnostic_slices(control).items() if mask.any()
        )
        specimen_diagnostics = tuple(
            ({**specimen, 'scope': 'specimen_diagnostic', 'slice': diagnostic['slice']}, positions[mask[positions]])
            for specimen, positions in groups if specimen['scope'] == 'specimen'
            for diagnostic, mask in diagnostics if mask[positions].any()
        )
        error_slices = groups + diagnostics + specimen_diagnostics
        for arm, frame in arms.items():
            decision = frame.prediction.to_numpy()
            names = {'feature_set': feature_set, 'arm': arm}
            for scope, indices in groups:
                matrix = _confusion(truth[indices], decision[indices])
                confusion.extend({**names, **scope, 'label': label,
                                  **{f'predicted_{p}': count for p, count in enumerate(counts)}}
                                 for label, counts in enumerate(matrix))
            if arm in (POINT_ONLY, 'prior_native'):
                continue
            references = [POINT_ONLY]
            if 'prior_rescue' in arms and arm != 'prior_rescue':
                references.append('prior_rescue')
            matched = [ref for tested, ref in MATCHED_PAIRS if arm == tested and ref in arms]
            for ref in (*references, *matched):
                comparison = {**names, 'reference_arm': ref}
                reference = arms[ref].prediction.to_numpy()
                changes.extend(_error_changes(truth, decision, reference, error_slices, comparison))
                if ref in matched:
                    matched_intervals.extend(
                        {**comparison, **result}
                        for result in _bootstrap(frame.assign(baseline_prediction=reference), draws, seed)
                    )
    tables['confusion'] = pd.DataFrame(confusion)
    tables['paired_error_changes'] = pd.DataFrame(changes)
    if matched_intervals:
        tables['paired_bootstrap'] = pd.concat(
            [tables['paired_bootstrap'], pd.DataFrame(matched_intervals)], ignore_index=True,
        )
    return tables


def _read_run(directory, required_arms):
    run_path = directory / 'run.json'
    plan_path = directory / 'plan.json'
    predictions_path = directory / 'predictions.parquet'
    run = read_json(run_path)
    digest = sha256_file(predictions_path)
    plan_digest = sha256_file(plan_path)
    archived = directory / 'mlartifacts' / run['run_id'] / 'artifacts'
    if run.get('predictions_sha256', digest) != digest:
        raise ValueError(f'Completed run checksum does not match {predictions_path}')
    for path, local_digest in ((predictions_path, digest), (plan_path, plan_digest)):
        logged = archived / path.name
        if not logged.is_file() or sha256_file(logged) != local_digest:
            raise ValueError(f'The completed MLflow run does not verify {path}')
    predictions = pd.read_parquet(predictions_path)
    if predictions.empty:
        raise ValueError(f'No held-out decisions in {directory}')
    for feature_set, frame in predictions.groupby('feature_set', sort=True):
        if set(frame.arm) != required_arms:
            raise ValueError(f'Unexpected arms for {feature_set} in {directory}')
    return predictions, {
        'run_id': run['run_id'],
        'predictions': str(predictions_path),
        'predictions_sha256': digest,
        'plan_sha256': plan_digest,
    }


def _new_axis_decisions(axis_predictions, fixed_predictions):
    features = axis_predictions.feature_set.unique()
    expected = fixed_predictions.loc[
        fixed_predictions.feature_set.isin(features) & fixed_predictions.arm.isin(AXIS_ARMS - {AXIS_ARM}), REQUIRED
    ]
    comparators = axis_predictions.loc[axis_predictions.arm.ne(AXIS_ARM), REQUIRED]
    pd.testing.assert_frame_equal(
        expected.sort_values(KEYS, ignore_index=True),
        comparators.sort_values(KEYS, ignore_index=True),
        check_dtype=False, obj='Axis-ablation archived comparators',
    )
    return axis_predictions.loc[axis_predictions.arm.eq(AXIS_ARM)]


def _plot_case_errors(path, predictions):
    """Map errors on specimens identified in the earlier postprocessing analysis."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    cases = predictions.loc[predictions.feature_set.eq('manual+tsfel')
                            & predictions.specimen_id.isin(CASE_BODIES)]
    titles = {POINT_ONLY: 'Punkt-SVM', 'prior_rescue': 'Nachbarschaftsrettung',
              'selected': 'Räumliche Auswahl', 'hpo_selected': 'Auswahl mit HPO'}
    arms = [arm for arm in titles if arm in set(cases.arm)]
    if not arms or set(cases.specimen_id) != set(CASE_BODIES):
        return
    markers = (
        ('#a5a9ae', '.', 'Übrige Messpunkte'),
        ('#9b202b', 'o', 'Hohlraum freigegeben (1→3)'),
        ('#c57111', '^', 'Unsicherer Punkt freigegeben (2→3)'),
        ('#2058a8', 'x', 'Fehlalarm (3→1/2)'),
    )
    figure = Figure(figsize=(2.9 * len(arms), 5.8), layout='constrained')
    axes = figure.subplots(len(CASE_BODIES), len(arms), sharex='row', sharey='row', squeeze=False)
    for row, body in enumerate(CASE_BODIES):
        for column, arm in enumerate(arms):
            axis = axes[row, column]
            frame = cases.loc[cases.specimen_id.eq(body) & cases.arm.eq(arm)]
            cavity = frame.label.eq(1) & frame.prediction.eq(3)
            uncertain = frame.label.eq(2) & frame.prediction.eq(3)
            false_alarm = frame.label.eq(3) & frame.prediction.isin([1, 2])
            masks = (~(cavity | uncertain | false_alarm), cavity, uncertain, false_alarm)
            for mask, (color, marker, _) in zip(masks, markers, strict=True):
                points = frame.loc[mask, ['x', 'y']]
                axis.scatter(points.x, points.y, s=9, c=color, marker=marker, linewidths=.7)
            axis.set_title(f'{titles[arm]}\n{(cavity | uncertain).sum()} kritische Freigaben', fontsize=9)
            axis.set_aspect('equal')
            axis.set_xlabel('X')
        axes[row, 0].set_ylabel(f'{body}\nY')
    handles = [Line2D([], [], c=color, marker=marker, linestyle='none', label=label)
               for color, marker, label in markers]
    figure.legend(handles=handles, loc='outside lower center', ncols=2, frameon=False)
    figure.suptitle('Manuell + TSFEL auf zwei vorab bekannten schwierigen Prüfkörpern')
    figure.savefig(path, dpi=170)


def run(fixed, output, *, hpo=None, axis=None, point_comparator=None):
    """Write the final combined evaluation in a fresh results directory."""
    fixed, output = Path(fixed).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Review already exists in {output}')
    predictions, source = _read_run(fixed, FIXED_ARMS)
    controls = predictions.loc[predictions.arm.eq(POINT_ONLY)]
    for _, cohort in controls.groupby('feature_set'):
        n_specimens = cohort.groupby('fold').specimen_id.nunique().to_dict()
        n_rows = cohort.groupby('fold').size().to_dict()
        if n_specimens != {0: 8, 1: 8, 2: 7} or n_rows != {0: 11975, 1: 11841, 2: 10419}:
            raise ValueError('Expected the frozen 23 training specimens and all outer predictions')
    if set(controls.feature_set) != set(read_json(fixed / 'plan.json')['feature_sets']):
        raise ValueError('Expected all frozen point feature sets')
    for feature_set, frame in predictions.groupby('feature_set'):
        arm_labels = frame.pivot(index='instance_id', columns='arm', values='prediction')
        if not arm_labels[POINT_ONLY].equals(arm_labels.prior_native):
            raise ValueError(f'Historical native SVM differs from the point control for {feature_set}')
    manifests = {'fixed': source}
    sources = [fixed]
    for name, directory, required in (('hpo', hpo, HPO_ARMS),
                                      ('point_comparator', point_comparator, {POINT_HPO})):
        if directory is None:
            continue
        directory = Path(directory).resolve()
        tuned, manifests[name] = _read_run(directory, required)
        predictions = pd.concat([predictions, tuned], ignore_index=True)
        sources.append(directory)
    if axis is not None:
        axis = Path(axis).resolve()
        directional, manifests['axis'] = _read_run(axis, AXIS_ARMS)
        new = _new_axis_decisions(directional, predictions)
        predictions = pd.concat([predictions, new], ignore_index=True)
        sources.append(axis)
    if any(output.is_relative_to(source) for source in sources):
        raise ValueError('Write the combined review outside prior experiment directories')
    tables = review_predictions(predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.spatial-review-') as temp:
        staged = Path(temp) / 'review'
        staged.mkdir()
        for name, table in tables.items():
            table.to_csv(staged / f'{name}.csv', index=False)
        _plot_case_errors(staged / 'specimen_error_maps.png', predictions)
        write_json(staged / 'manifest.json', {
            'protocol': 'spatial_svm_outer_review_v1',
            'inputs': manifests,
            'draws': 2000,
            'seed': 42,
            'code_sha256': {
                path.name: sha256_file(path)
                for path in (Path(__file__), Path(__file__).with_name('spatial_feature_report.py'),
                             Path(__file__).with_name('spatial_postprocessing_report.py'))
            },
            'report_sha256': {path.name: sha256_file(path) for path in sorted(staged.iterdir())},
        })
        staged.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed', type=Path, required=True)
    parser.add_argument('--hpo', type=Path)
    parser.add_argument('--axis', type=Path)
    parser.add_argument('--point-comparator', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.fixed, args.output_dir, hpo=args.hpo, axis=args.axis, point_comparator=args.point_comparator)


if __name__ == '__main__':
    main()
