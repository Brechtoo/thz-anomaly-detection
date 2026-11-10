"""Portable body-fold metadata must describe the existing training cohort exactly."""
import json

import pandas as pd

from src.data.split_manifest import sha256_file
from src.paths import COMBINED_DATASET_ROOT, MEASUREMENTS_ROOT, PROJECT_ROOT


def test_frozen_body_folds_cover_only_original_training_bodies():
    folder = PROJECT_ROOT / 'configs/anchor_group_cv_v1'
    manifest = json.loads((folder / 'manifest.json').read_text())
    folds = pd.read_csv(folder / 'specimen_folds.csv')
    assert sha256_file(folder / 'specimen_folds.csv') == manifest['files']['specimen_folds.csv']
    assert sha256_file(PROJECT_ROOT / manifest['train_split']) == manifest['train_split_sha256']
    membership = pd.read_csv(PROJECT_ROOT / 'configs/specimen_holdout_v1/splits/membership.csv')
    train = membership[membership.split == 'train']
    assert folds.specimen_id.is_unique and len(folds) == 23
    assert set(folds.specimen_id) == set(train.specimen_id)
    combined = train.merge(folds, on='specimen_id', validate='many_to_one')
    assert combined.instance_id.is_unique and len(combined) == 34235
    assert {str(k): int(v) for k, v in combined.groupby('fold').size().items()} == manifest['points_per_fold']
    assert {str(k): int(v) for k, v in folds.groupby('fold').size().items()} == manifest['bodies_per_fold']
    assert set(combined.specimen_id).isdisjoint(set(membership.specimen_id[membership.split != 'train']))


def test_catalog_defaults_use_active_data_directory():
    assert MEASUREMENTS_ROOT == PROJECT_ROOT / 'data/measurements'
    assert COMBINED_DATASET_ROOT == MEASUREMENTS_ROOT / 'combined_dataset.csv'
