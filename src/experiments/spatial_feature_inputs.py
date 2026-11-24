"""Verify archived inputs for the spatial-feature SVM experiment."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split_manifest import sha256_file
from src.evaluation.svm_nested_selection import LABELS
from src.experiments.spatial_inputs import (
    KEYS,
    _frozen_cohort,
    _inner_partitions,
    _same_rows,
)
from src.experiments.svm_nested_hpo import FEATURES, NestedFeatures
from src.features.specimen_grid import TABLE_FILES
from src.features.specimen_inputs import ANCHOR, EARLY, read_json
from src.paths import FEATURE_TABLES_ROOT, PROJECT_ROOT


@dataclass(frozen=True)
class SpatialFeatureInputs:
    train: pd.DataFrame
    assignments: pd.DataFrame
    features: NestedFeatures
    plan: dict
    hashes: dict[str, str]


def _record(hashes, path, expected=None):
    key = str(path.resolve())
    if key not in hashes:
        hashes[key] = sha256_file(path)
    if expected is not None and hashes[key] != expected:
        raise ValueError(f'Checksum mismatch: {path}')
    return hashes[key]


def _archived_digest(digests, path, root):
    # Archived plans name files by absolute path. The suffix survives a project move.
    suffix = '/' + path.relative_to(root).as_posix()
    matches = [digest for name, digest in digests.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one archived checksum for {suffix}')
    return matches[0]


def load_spatial_feature_inputs(nested_dir):
    """Load the same training rows, inner folds and feature blocks as nested HPO."""
    nested_dir = Path(nested_dir).resolve()
    hashes = {}

    def record(path, expected=None):
        return _record(hashes, path, expected)

    plan_path = nested_dir / 'plan.json'
    record(plan_path)
    plan = read_json(plan_path)
    if (plan['protocol'] != 'nested_train_only_v1' or plan['class_order'] != LABELS.tolist()
            or plan['features'] != {name: list(blocks) for name, blocks in FEATURES.items()}):
        raise ValueError('Unexpected nested-SVM experiment plan')
    source_hashes = plan['source_sha256']
    train = _frozen_cohort(record, source_hashes).assign(split='train')
    # The manual NPZ is positional. Its extraction cohort must already use exactly
    # the canonical row order, before any X/Y join or tabular preprocessing.
    cohort_path = ANCHOR / 'cohort_folds.csv'
    if not pd.read_csv(cohort_path, usecols=['instance_id']).instance_id.equals(train.instance_id):
        raise ValueError('Manual feature rows disagree with the canonical acquisition order')
    assignments_path = nested_dir / 'inner_assignments.parquet'
    record(assignments_path, plan['inner_assignment_sha256'])
    assignments = pd.read_parquet(assignments_path)
    if set(assignments.outer) != set(train.fold):
        raise ValueError('Unexpected outer folds in frozen inner assignments')
    for outer in sorted(train.fold.unique()):
        allowed = _inner_partitions(train, assignments, outer)
        for inner in sorted(allowed.fold.unique()):
            if set(allowed.loc[allowed.fold.eq(inner), 'label']) != set(LABELS):
                raise ValueError(f'Missing class in inner fold {outer}/{inner}')

    extraction = ANCHOR / 'extraction_manifest.json'
    tables = [FEATURE_TABLES_ROOT / TABLE_FILES[name] for name in ('tsfel', 'physical_v2')]
    for path in (extraction, ANCHOR / 'plan.json', *(p.with_suffix('.json') for p in tables)):
        record(path, _archived_digest(source_hashes, path, PROJECT_ROOT))
    manual = ANCHOR / 'features' / f'{EARLY}.npz'
    manifest = read_json(extraction)
    record(cohort_path, manifest['cohort_sha256'])
    record(manual, manifest['features_sha256'][manual.name])
    for path in tables:
        record(path, read_json(path.with_suffix('.json'))['parquet_sha256'])
    return SpatialFeatureInputs(train, assignments, NestedFeatures(train), plan, hashes)


def load_spatial_feature_reference(nested_dir, post_dir, train, baselines):
    """Read the earlier postprocessing decisions as a paired, held-out reference."""
    nested_dir, post_dir = Path(nested_dir).resolve(), Path(post_dir).resolve()
    hashes = {}
    run_path = post_dir / 'run.json'
    _record(hashes, run_path)
    run = read_json(run_path)
    if not re.fullmatch('[0-9a-f]{32}', run['run_id']):
        raise ValueError('Unexpected postprocessing run ID')
    logged = post_dir / 'mlartifacts' / run['run_id'] / 'artifacts'
    for filename in ('plan.json', 'predictions.parquet'):
        _record(hashes, post_dir / filename, _record(hashes, logged / filename))
    plan = read_json(post_dir / 'plan.json')
    # Match the prior experiment's plan and partitions to this archival source.
    if plan['protocol'] != 'spatial_nested_train_oof_v1' or not set(baselines) <= set(plan['baselines']):
        raise ValueError('Unexpected archived spatial postprocessing plan')
    input_hashes = plan['inputs_sha256']
    for path in (nested_dir / 'plan.json', nested_dir / 'inner_assignments.parquet'):
        _record(hashes, path, _archived_digest(input_hashes, path, nested_dir.parent))
    columns = [*KEYS, 'x', 'y', 'baseline', 'strategy', 'prediction', 'baseline_prediction']
    prior = pd.read_parquet(post_dir / 'predictions.parquet', columns=columns)
    output = []
    for feature_set in baselines:
        native = []
        for fold, held in train.groupby('fold', sort=True):
            directory = nested_dir / 'outer' / feature_set / f'fold_{fold}'
            provenance = directory / 'provenance.json'
            predictions = directory / 'predictions.parquet'
            _record(hashes, provenance, _archived_digest(input_hashes, provenance, nested_dir.parent))
            digest = _record(hashes, predictions, read_json(provenance)['sha256'])
            if digest != _archived_digest(input_hashes, predictions, nested_dir.parent):
                raise ValueError(f'Nested predictions differ from postprocessing input: {predictions}')
            saved = pd.read_parquet(predictions, columns=[*KEYS, 'prediction_regular'])
            native.append(_same_rows(saved, held, predictions))
        reference = pd.concat(native).set_index('instance_id').loc[train.instance_id, 'prediction_regular']

        for strategy, arm in (('baseline', 'prior_native'), ('rescue', 'prior_rescue')):
            frame = prior.loc[prior.baseline.eq(feature_set) & prior.strategy.eq(strategy)]
            frame = _same_rows(frame, train, f'{feature_set}/{strategy}')
            if (not np.array_equal(frame[['x', 'y']], train[['x', 'y']])
                    or not frame.prediction.isin(LABELS).all()
                    or not np.array_equal(frame.baseline_prediction, reference)
                    or (strategy == 'baseline' and not np.array_equal(frame.prediction, reference))):
                raise ValueError(f'Archived postprocessing predictions do not match {feature_set}/{strategy}')
            output.append(frame[[*KEYS, 'x', 'y', 'prediction']].assign(feature_set=feature_set, arm=arm))
    return pd.concat(output, ignore_index=True), hashes
