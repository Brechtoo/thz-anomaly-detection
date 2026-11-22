"""Assemble early-anchor manual and tsfresh features on frozen specimen cohorts."""
import hashlib

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.features.specimen_inputs import require_hash
from src.paths import EXPERIMENTS_ROOT

MANUAL_CACHE = EXPERIMENTS_ROOT / 'svm_group_cv_manual52_v1'
HOLDOUT_HASHES = {
    'early_holdout_features.csv': '1096ca02f48924bedb1f6118e48c01031ebb07e1a56b845f850014ce3c825218',
    'early_holdout_features.npz': '4fdbe33c80be275bae3dbbf3a40ed3fc61cc127eb23edd1ba34c5c27d49c1978',
}


def load_manual_holdout(membership, train_features):
    holdout = membership.loc[membership.split.ne('train')]
    for name, digest in HOLDOUT_HASHES.items():
        require_hash(MANUAL_CACHE / name, digest)
    ids = pd.read_csv(MANUAL_CACHE / 'early_holdout_features.csv').instance_id
    if not ids.equals(holdout.instance_id.reset_index(drop=True)):
        raise ValueError('Early manual holdout IDs disagree with frozen membership')
    with np.load(MANUAL_CACHE / 'early_holdout_features.npz') as source:
        features = pd.DataFrame(source['X'], index=holdout.index, columns=train_features.columns)
    return pd.concat([train_features, features]).loc[membership.index]


def fuse_features(manual, tsfresh, train_index):
    """Learn column filtering on fit rows, retaining manual columns first."""
    if not manual.index.equals(tsfresh.index) or not manual.index.is_unique:
        raise ValueError('Fusion requires exactly aligned measurement indices')
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    imputer.fit(manual.loc[train_index])
    values = pd.DataFrame(
        imputer.transform(manual), index=manual.index, columns=manual.columns,
    ).add_prefix('manual__')
    combined = pd.concat([values, tsfresh.add_prefix('tsfresh__')], axis=1)

    training = combined.loc[train_index]
    if training.empty or not training.columns.is_unique or not np.isfinite(combined.to_numpy()).all():
        raise ValueError('Fusion requires a finite matrix and unique feature names')
    constants, duplicates, representatives = [], {}, {}
    # Manual columns come first, so equivalent tsfresh columns are discarded in their favor.
    keep = []
    for name, column in training.items():
        data = column.to_numpy(dtype=np.float64, copy=True)
        if (data == data[0]).all():
            constants.append(name)
            continue
        data[data == 0] = 0.0  # IEEE signed zeros are numerically equal
        signature = hashlib.sha256(data).digest()
        peers = representatives.setdefault(signature, [])
        original = next((peer for peer in peers if np.array_equal(data, training[peer])), None)
        if original is not None:
            duplicates[name] = original
        else:
            peers.append(name)
            keep.append(name)
    return combined.loc[:, keep], {
        'manual_medians': dict(zip(values.columns, imputer.statistics_.tolist())),
        'tsfresh_selected': tsfresh.columns.tolist(),
        'removed_constants': constants, 'removed_duplicates': duplicates,
        'retained_features': keep,
    }
