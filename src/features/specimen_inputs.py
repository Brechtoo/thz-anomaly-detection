"""Verified feature inputs and fixed cohorts shared by specimen experiments."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.specimen_splits import load_membership
from src.data.split_manifest import sha256_file
from src.features.experimental_anchors import extract_matrix
from src.paths import EXPERIMENTS_ROOT, FEATURE_TABLE_ROOT, PROJECT_ROOT

SPLITS = PROJECT_ROOT / 'configs/specimen_holdout_v1/splits'
FOLDS = PROJECT_ROOT / 'configs/anchor_group_cv_v1'
ANCHOR = EXPERIMENTS_ROOT / 'anchor_group_cv_v1'
TSFRESH = EXPERIMENTS_ROOT / 'specimen_holdout_v1/features'
EARLY = 'early_legacy_windows'


def require_hash(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f'Checksum mismatch: {path}')


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_cohorts():
    membership = load_membership(SPLITS).sort_values('instance_id').reset_index(drop=True)
    manifest = read_json(FOLDS / 'manifest.json')
    fold_path = FOLDS / 'specimen_folds.csv'
    require_hash(fold_path, manifest['files'][fold_path.name])
    require_hash(SPLITS / 'train.csv', manifest['train_split_sha256'])
    assignment = pd.read_csv(fold_path).set_index('specimen_id', verify_integrity=True)['fold']
    train = membership.loc[membership.split.eq('train')].copy()
    if set(assignment.index) != set(train.specimen_id):
        raise ValueError('Fold assignment must cover exactly the training bodies')
    train['fold'] = train.specimen_id.map(assignment)
    if train.fold.value_counts().sort_index().to_dict() != {
        int(k): v for k, v in manifest['points_per_fold'].items()
    }:
        raise ValueError('Unexpected fold sizes')
    historic = ANCHOR / 'cohort_folds.csv'
    require_hash(historic, manifest['source_sha256'])
    pd.testing.assert_frame_equal(
        train.reset_index(drop=True), pd.read_csv(historic, usecols=[
            'instance_id', 'specimen_id', 'source_dataset', 'label', 'fold',
        ]).assign(split='train')[train.columns],
    )
    return membership, train


def tsfresh_manifest():
    manifest = read_json(TSFRESH / 'manifest.json')
    inputs = TSFRESH / 'extraction_inputs.json'
    require_hash(inputs, manifest['input_sha256'])
    require_hash(SPLITS / 'manifest.json', read_json(inputs)['split_sha256'])
    return manifest


def raw_tsfresh(membership, train, manifest):
    raw_ids = pd.read_csv(TSFRESH / 'raw_inputs.csv', usecols=['instance_id'])
    if not raw_ids.instance_id.equals(membership.instance_id):
        raise ValueError('Raw batch order disagrees with specimen membership')
    batches = []
    names = None
    for start in range(0, len(membership), 512):
        path = TSFRESH / 'raw_batches' / f'{start:06d}.npz'
        require_hash(path, manifest['raw_batches_sha256'][path.name])
        with np.load(path) as batch:
            columns = batch['names'].tolist()
            if names is not None and names != columns:
                raise ValueError('Inconsistent raw feature order')
            names = columns
            expected = min(512, len(membership) - start)
            if batch['values'].shape != (expected, manifest['raw_features']):
                raise ValueError('Unexpected raw feature batch shape')
            positions = train.index[(train.index >= start) & (train.index < start + expected)]
            batches.append(batch['values'][positions - start])
    if len(set(names)) != len(names) or not all(name.startswith('y__') for name in names):
        raise ValueError('Unexpected tsfresh feature schema')
    return pd.DataFrame(np.concatenate(batches), index=train.index, columns=names)


def early_holdouts(output, membership, train_features, columns, workers):
    """Extend the frozen early-anchored cache using the same signal-wise extractor."""
    holdout = membership.loc[~membership.split.eq('train')].copy()
    raw = pd.read_csv(TSFRESH / 'raw_inputs.csv').set_index('instance_id', verify_integrity=True)
    raw = raw.loc[holdout.instance_id]
    expected = read_json(TSFRESH / 'extraction_inputs.json')['raw_hashes']
    if list(raw.sha256) != [expected[i] for i in holdout.index]:
        raise ValueError('Holdout waveform hashes disagree with archived extraction inputs')
    for path, digest in raw[['file_path', 'sha256']].itertuples(index=False, name=None):
        require_hash(Path(path), digest)
    holdout['file_path'] = raw.file_path.to_numpy()
    source_hash = read_json(PROJECT_ROOT / 'configs/specimen_holdout_v1/manual_features.json')
    require_hash(FEATURE_TABLE_ROOT, source_hash['source_table_sha256'])
    base = pd.read_csv(FEATURE_TABLE_ROOT, usecols=['instance_id', *columns[:17]])
    base = base.set_index('instance_id', verify_integrity=True).loc[holdout.instance_id, columns[:17]]
    inputs = np.full((len(holdout), len(columns)), np.nan)
    inputs[:, :17] = base.to_numpy()
    path = output / 'early_holdout_features.npz'
    with ProcessPoolExecutor(max_workers=workers) as pool:
        extract_matrix(holdout, inputs, EARLY, columns[17:], None, path, pool)
    with np.load(path) as saved:
        holdout_features = pd.DataFrame(saved['X'], index=holdout.index, columns=columns)
    return pd.concat([train_features, holdout_features]).loc[membership.index]


def load_tsfresh_holdout(membership, manifest):
    path = TSFRESH / 'features.csv'
    require_hash(path, manifest['features_sha256'])
    table = pd.read_csv(path).set_index('instance_id', verify_integrity=True).loc[membership.instance_id]
    if not np.array_equal(table.pop('label'), membership.label):
        raise ValueError('tsfresh labels differ from frozen membership')
    if table.columns.tolist() != manifest['selected_features']:
        raise ValueError('Unexpected holdout feature schema')
    return table.set_axis(membership.index)



def load_early_manual_train(train):
    manifest = read_json(ANCHOR / 'extraction_manifest.json')
    path = ANCHOR / 'features' / f'{EARLY}.npz'
    require_hash(path, manifest['features_sha256'][path.name])
    require_hash(ANCHOR / 'plan.json', manifest['plan_sha256'])
    columns = read_json(ANCHOR / 'plan.json')['features']
    with np.load(path) as saved:
        return pd.DataFrame(saved['X'], index=train.index, columns=columns)
