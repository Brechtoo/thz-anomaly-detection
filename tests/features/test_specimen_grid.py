from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from src.data.split_manifest import sha256_file
from src.features import specimen_grid
from src.features.specimen_grid import GridFeatures, verified_table
from src.features.specimen_inputs import write_json


def cohort():
    membership = pd.DataFrame({
        'instance_id': ['a', 'b', 'c', 'd', 'e', 'f'],
        'specimen_id': ['A', 'A', 'B', 'B', 'C', 'D'],
        'source_dataset': ['series1'] * 6,
        'label': [0, 1, 2, 3, 0, 3],
        'split': ['train', 'train', 'train', 'train', 'validation', 'test'],
    })
    train = membership.iloc[:4].assign(fold=[0, 0, 1, 1])
    return membership, train


def test_verified_table_realigns_shuffled_rows_and_rejects_wrong_fold(tmp_path):
    membership, train = cohort()
    features = membership.assign(
        cv_fold=train.fold.reindex(membership.index).astype('Int64'),
        example=np.arange(len(membership), dtype=float),
    )
    path = tmp_path / 'features.parquet'

    def save(rows):
        rows.to_parquet(path, index=False)
        write_json(path.with_suffix('.json'), {
            'metadata_columns': [*membership.columns, 'cv_fold'],
            'retained_features': ['example'],
            'parquet_sha256': sha256_file(path),
        })

    save(features.iloc[::-1])
    actual = verified_table(path, membership, train, expected_width=1)
    pd.testing.assert_frame_equal(actual, features[['example']])
    features.loc[0, 'cv_fold'] = 1
    save(features)
    with pytest.raises(AssertionError):
        verified_table(path, membership, train, expected_width=1)


def test_fold_only_medians_and_exact_redundancy():
    grid = GridFeatures.__new__(GridFeatures)
    grid.membership, grid.train = cohort()
    grid.frames = {
        'manual': pd.DataFrame({'base': [100, 200, 1, 2], 'observed_constant': [49, 48, 7, 7]}),
        'tsfel': pd.DataFrame({'duplicate': [0, 0, 1, 2, 999, 999],
                               'different': [np.nan, 1500, 12, 16, 999, 999]}),
    }
    grid._blocks = lru_cache(maxsize=20)(grid._prepare_block)
    X, Z, state = grid.fold(0, ('tsfel', 'manual'))
    assert X.flags.c_contiguous and Z.flags.c_contiguous
    np.testing.assert_array_equal(X, [[1, 12], [2, 16]])
    np.testing.assert_array_equal(Z, [[100, 14], [200, 1500]])
    assert state['removed_constants'] == ['manual::observed_constant']
    assert state['removed_duplicates'] == {'tsfel::duplicate': 'manual::base'}
    assert state['retained_features'] == ['manual::base', 'tsfel::different']
    assert state['preprocessing_by_set']['tsfel']['medians']['different'] == 14
    with pytest.raises(ValueError, match='Invalid feature combination'):
        grid.fold(0, ('physical_v1', 'physical_v2'))


def test_tsfresh_cache_refits_on_fit_rows_only_and_checks_checksum(tmp_path, monkeypatch):
    grid = GridFeatures.__new__(GridFeatures)
    grid.membership, grid.train = cohort()
    grid.cache_dir = tmp_path
    grid.source_fingerprints = {'fixture': 'hash'}
    grid.tsfresh_versions = {}
    grid.raw = pd.DataFrame({'y__raw': [11.0, 12.0, 13.0, 14.0]})
    calls = []

    def select(features, labels, train_index, *, return_ranges):
        assert return_ranges
        calls.append((list(train_index), list(labels.loc[train_index])))
        return features, tuple({'y__raw': 13.0} for _ in range(3))

    monkeypatch.setattr(specimen_grid, 'select_training_features', select)
    mask = grid.train.fold.ne(0)
    first, columns, info = grid._fold_tsfresh(0, mask)
    second, _, _ = grid._fold_tsfresh(0, mask)
    assert calls == [([2, 3], [2, 3])]
    assert columns == ['y__raw']
    assert 'refitted in the fold' in info['method']
    np.testing.assert_array_equal(first, second)
    assert not second.flags.writeable
    with (tmp_path / 'tsfresh_fold_0/values.npy').open('ab') as stream:
        stream.write(b'invalidated')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        grid._fold_tsfresh(0, mask)
