"""Read and verify the training-only inputs of the spatial experiment."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.reference_measurements import require_no_references
from src.data.split_manifest import sha256_file
from src.evaluation.leakage import ACQUISITION_PATTERN
from src.evaluation.svm_nested_selection import (
    CALIBRATIONS,
    LABELS,
    SCORE_COLUMNS,
    body_cross_fitted_probabilities,
    fit_calibrator,
    predict_probabilities,
)
from src.features.specimen_inputs import ANCHOR, FOLDS, SPLITS, read_json
from src.paths import PROJECT_ROOT

KEYS = ['instance_id', 'specimen_id', 'source_dataset', 'label', 'fold']
PROBABILITIES = [f'p_{label}' for label in LABELS]


@dataclass(frozen=True)
class SpatialFold:
    fold: int
    inner: pd.DataFrame
    outer: pd.DataFrame
    candidate: str
    calibration: str


def _same_rows(frame, expected, context):
    """Check exact membership and return the artifact in expected ID order."""
    index = 'instance_id'
    if (frame[index].isna().any() or not frame[index].is_unique
            or set(frame[index]) != set(expected[index])):
        raise ValueError(f'Prediction ID coverage differs in {context}')
    aligned = frame.set_index(index).loc[expected[index]].reset_index()
    try:
        pd.testing.assert_frame_equal(
            aligned[KEYS], expected[KEYS].reset_index(drop=True), check_dtype=False,
        )
    except AssertionError as error:
        raise ValueError(f'Prediction metadata differs in {context}') from error
    return aligned


def _attach_coordinates(cohort):
    """Use the recorded XY positions in the training-only acquisition filenames."""
    names = cohort.file_path.map(lambda name: Path(name).name)
    coordinates = names.str.extract(ACQUISITION_PATTERN).coordinates.str.split(',', expand=True)
    if coordinates.shape[1] != 3:
        raise ValueError('Unexpected acquisition coordinate format')
    xy = coordinates.iloc[:, :2].apply(pd.to_numeric).to_numpy(dtype=float)
    ids = cohort.instance_id.str.extract(r'^(?P<source>messreihe_[123])_x(?P<x>-?\d+)_y(?P<y>-?\d+)$')
    if (not np.isfinite(xy).all() or ids.isna().any().any()
            or not ids.source.equals(cohort.source_dataset)
            or not np.array_equal(np.rint(xy), ids[['x', 'y']].to_numpy(dtype=int))):
        raise ValueError('Acquisition coordinates disagree with the frozen measurement IDs')
    return cohort.drop(columns='file_path').assign(x=xy[:, 0], y=xy[:, 1])


def _frozen_cohort(record, source_hashes):
    manifest_path = FOLDS / 'manifest.json'
    train_path = SPLITS / 'train.csv'
    for path in (manifest_path, train_path):
        # The nested plan stores absolute paths. Use the project-relative suffix
        # to verify canonical inputs after relocating a checkout.
        suffix = '/' + path.relative_to(PROJECT_ROOT).as_posix()
        matches = [sha for name, sha in source_hashes.items() if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f'Canonical input is missing from the source plan: {suffix}')
        record(path, matches[0])
    manifest = read_json(manifest_path)
    record(train_path, manifest['train_split_sha256'])
    fold_path = FOLDS / 'specimen_folds.csv'
    record(fold_path, manifest['files'][fold_path.name])
    anchor_path = ANCHOR / 'cohort_folds.csv'
    record(anchor_path, manifest['source_sha256'])
    cohort = pd.read_csv(anchor_path, usecols=[*KEYS, 'file_path']).sort_values('instance_id').reset_index(drop=True)
    training = pd.read_csv(train_path).sort_values('instance_id').reset_index(drop=True)
    if not cohort[['instance_id', 'label']].equals(training[['instance_id', 'label']]):
        raise ValueError('Anchor cohort differs from the frozen train split')
    if cohort[[*KEYS, 'file_path']].isna().any(axis=None) or not cohort.instance_id.is_unique:
        raise ValueError('Incomplete or duplicated train metadata')
    assignment = pd.read_csv(fold_path).set_index('specimen_id')['fold']
    groups = cohort[['specimen_id', 'source_dataset', 'fold']].drop_duplicates()
    if (not groups.specimen_id.is_unique or not assignment.index.is_unique
            or set(groups.specimen_id) != set(assignment.index)
            or not cohort.specimen_id.map(assignment).eq(cohort.fold).all()
            or set(cohort.fold) != {0, 1, 2}
            or cohort.fold.value_counts().to_dict() != {int(key): n for key, n in manifest['points_per_fold'].items()}
            or groups.fold.value_counts().to_dict() != {int(key): n for key, n in manifest['bodies_per_fold'].items()}
            or set(cohort.label) != set(LABELS)):
        raise ValueError('Unexpected frozen training group assignments')
    require_no_references(cohort)
    return _attach_coordinates(cohort)


def _inner_partitions(cohort, assignments, outer):
    allowed = cohort.loc[cohort.fold.ne(outer)].copy()
    partition = assignments.loc[assignments.outer.eq(outer)]
    original = partition.rename(columns={'fold': 'inner_fold', 'original_fold': 'fold'})
    _same_rows(original, allowed, f'inner assignments for outer {outer}')
    inner_folds = partition.set_index('instance_id').fold
    allowed['fold'] = allowed.instance_id.map(inner_folds)
    if set(allowed.fold) != {0, 1, 2} or allowed.groupby('specimen_id').fold.nunique().ne(1).any():
        raise ValueError(f'Invalid inner group partitions for outer {outer}')
    return allowed


def _check_predictions(frame):
    values = frame[PROBABILITIES].to_numpy()
    if (not frame.prediction.isin(LABELS).all() or not np.isfinite(values).all()
            or not ((values >= 0) & (values <= 1)).all()
            or not np.allclose(values.sum(axis=1), 1., rtol=0., atol=1e-10)):
        raise ValueError('Invalid four-class predictions')


def load_spatial_input(source: Path, features: str) -> tuple[list[SpatialFold], dict[str, str]]:
    """Reuse regular nested-HPO predictions and cross-fit inner probabilities.

    Outer decisions are the saved native SVM classes. Spatial tuning can use
    the corresponding inner validation predictions, with the spatial rule and
    the SVM search evaluated jointly on the untouched outer specimens.
    """
    hashes = {}

    def record(path, expected=None):
        path = path.resolve()
        if str(path) not in hashes:
            hashes[str(path)] = sha256_file(path)
        if expected is not None and hashes[str(path)] != expected:
            raise ValueError(f'Checksum mismatch: {path}')

    plan_path = source / 'plan.json'
    record(plan_path)
    plan = read_json(plan_path)
    if (plan['protocol'] != 'nested_train_only_v1' or plan['class_order'] != LABELS.tolist()
            or features not in plan['features'] or Path(features).name != features or features == '..'):
        raise ValueError('Expected a completed training-only nested-SVM feature set')
    candidate_names = {candidate['candidate'] for candidate in plan['candidates']}
    cohort = _frozen_cohort(record, plan['source_sha256'])
    assignments_path = source / 'inner_assignments.parquet'
    record(assignments_path, plan['inner_assignment_sha256'])
    assignments = pd.read_parquet(assignments_path)
    if set(assignments.outer) != {0, 1, 2}:
        raise ValueError('Invalid outer values in inner assignments')

    folds = []
    for outer in range(3):
        allowed = _inner_partitions(cohort, assignments, outer)
        held = cohort.loc[cohort.fold.eq(outer)]
        work = source / 'search' / features / f'outer_{outer}'
        selection_path = work / 'selection.json'
        record(selection_path)
        selected = read_json(selection_path)['regular']
        candidate = selected['candidate']
        method = selected['calibration']
        if (candidate not in candidate_names or Path(candidate).name != candidate or candidate == '..'
                or method not in CALIBRATIONS):
            raise ValueError(f'Unexpected regular selection in fold {outer}')

        inner_frames = []
        for inner, group in allowed.groupby('fold', sort=True):
            directory = work / f'inner_{inner}' / candidate
            fit_path, predictions_path = directory / 'fit.json', directory / 'predictions.parquet'
            record(fit_path)
            record(predictions_path, read_json(fit_path)['sha256'])
            prediction = _same_rows(pd.read_parquet(predictions_path), group, directory)
            prediction[['x', 'y']] = group[['x', 'y']].to_numpy()
            inner_frames.append(prediction)
        tuning = pd.concat(inner_frames, ignore_index=True)
        tuning[PROBABILITIES] = body_cross_fitted_probabilities(tuning, method)

        directory = source / 'outer' / features / f'fold_{outer}'
        provenance_path, predictions_path = directory / 'provenance.json', directory / 'predictions.parquet'
        record(provenance_path)
        provenance = read_json(provenance_path)
        record(predictions_path, provenance['sha256'])
        calibration = provenance['calibration']['regular']
        if calibration['method'] != method or calibration['inputs'] != len(tuning):
            raise ValueError(f'Outer calibration provenance differs in fold {outer}')
        outer_fit = work / 'outer_fits' / candidate / 'predictions.parquet'
        record(outer_fit, calibration['fit_times']['sha256'])
        original = _same_rows(pd.read_parquet(outer_fit), held, outer_fit)
        regular_columns = {f'regular_{column}': column for column in [*PROBABILITIES, *SCORE_COLUMNS]}
        saved = pd.read_parquet(predictions_path, columns=[*KEYS, 'prediction_regular', *regular_columns])
        evaluated = _same_rows(saved.rename(columns={**regular_columns, 'prediction_regular': 'prediction'}), held, directory)
        comparison = ['prediction', *SCORE_COLUMNS]
        if not evaluated[comparison].equals(original[comparison]):
            raise ValueError(f'Outer predictions differ from the selected native SVM in fold {outer}')
        # Reconstruct the fixed calibrator from the verified allowed scores.
        # The experiment still uses the stored regular probabilities.
        calibrator = fit_calibrator(tuning[SCORE_COLUMNS].to_numpy(), tuning.label.to_numpy(), method)
        reconstructed = predict_probabilities(calibrator, evaluated[SCORE_COLUMNS].to_numpy())
        if not np.allclose(evaluated[PROBABILITIES], reconstructed, rtol=0., atol=1e-8):
            raise ValueError(f'Stored probabilities disagree with training-only calibration in fold {outer}')
        evaluated[['x', 'y']] = held[['x', 'y']].to_numpy()
        _check_predictions(tuning)
        _check_predictions(evaluated)
        columns = [*KEYS, 'x', 'y', 'prediction', *PROBABILITIES]
        folds.append(SpatialFold(outer, tuning[columns], evaluated[columns], candidate, method))
    return folds, hashes
