"""Verified specimen inputs with shared, train-only preprocessing for feature grids."""
import hashlib
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.data.split_manifest import sha256_file
from src.features.extract_tsfresh_features import select_training_features
from src.features.specimen_fusion import (
    HOLDOUT_HASHES,
    MANUAL_CACHE,
    load_manual_holdout,
)
from src.features.specimen_inputs import (
    ANCHOR,
    FOLDS,
    SPLITS,
    TSFRESH,
    load_early_manual_train,
    load_tsfresh_holdout,
    raw_tsfresh,
    read_json,
    require_hash,
    tsfresh_manifest,
    write_json,
)
from src.paths import FEATURE_TABLES_ROOT, PROJECT_ROOT

INPUT_SETS = ('manual', 'tsfresh', 'tsfel', 'physical_v1', 'physical_v2')
TABLE_FILES = {
    'tsfel': 'specimen_tsfel_v1.parquet',
    'physical_v1': 'specimen_thz_physical_v1.parquet',
    'physical_v2': 'specimen_thz_physical_v2.parquet',
}
EXPECTED_WIDTH = {'manual': 52, 'tsfel': 72, 'physical_v1': 37, 'physical_v2': 39}


def verified_table(path, membership, train, expected_width):
    """Check hashes, schema and the complete ID/label/group/split/fold key."""
    manifest = read_json(path.with_suffix('.json'))
    require_hash(path, manifest['parquet_sha256'])
    table = pd.read_parquet(path)
    metadata = list(membership.columns) + ['cv_fold']
    columns = manifest['retained_features']
    if (manifest['metadata_columns'] != metadata or len(columns) != expected_width
            or len(set(columns)) != len(columns) or set(metadata) & set(columns)
            or table.columns.tolist() != metadata + columns):
        raise ValueError(f'Unexpected feature schema: {path}')
    if len(table) != len(membership) or not table.instance_id.is_unique or table.instance_id.isna().any():
        raise ValueError(f'Invalid measurement IDs: {path}')
    aligned = table.set_index('instance_id').reindex(membership.instance_id)
    aligned.index = membership.index
    pd.testing.assert_frame_equal(aligned[metadata[1:-1]], membership[metadata[1:-1]], check_exact=True)
    expected_folds = train.fold.reindex(membership.index).astype('Int64').rename('cv_fold')
    pd.testing.assert_series_equal(aligned.cv_fold.astype('Int64'), expected_folds, check_exact=True)
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in aligned[columns].dtypes):
        raise ValueError(f'Non-numeric feature column: {path}')
    return aligned[columns]


def training_column_signatures(array, fit_mask):
    """Find exact constants and order-independent signatures using only fit rows."""
    fit = array[fit_mask]
    if len(fit) < 2 or not np.isfinite(array).all():
        raise ValueError('Need two or more fit rows and finite imputed features')
    signatures = []
    for column in fit.T:
        if (column == column[0]).all():
            signatures.append(None)
        else:
            values = column.copy()
            values[values == 0] = 0.0
            signatures.append(hashlib.sha256(values).digest())
    return signatures


class GridFeatures:
    """Prepare data once per feature source and outer fold, then reuse it in the grid."""

    def __init__(self, membership, train, cache_dir):
        self.membership = membership
        self.train = train
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tsfresh_manifest = tsfresh_manifest()
        self.raw = None
        self.frames = {'manual': load_early_manual_train(train)}
        if self.frames['manual'].shape != (len(train), EXPECTED_WIDTH['manual']):
            raise ValueError('Expected the frozen early-anchored 52 manual features')
        for name, filename in TABLE_FILES.items():
            self.frames[name] = verified_table(FEATURE_TABLES_ROOT / filename, membership,
                                               train, EXPECTED_WIDTH[name])

        inputs = [SPLITS / 'manifest.json', FOLDS / 'manifest.json',
                  ANCHOR / 'extraction_manifest.json', ANCHOR / 'plan.json',
                  TSFRESH / 'manifest.json', TSFRESH / 'extraction_inputs.json',
                  *(FEATURE_TABLES_ROOT / filename.replace('.parquet', '.json')
                    for filename in TABLE_FILES.values()),
                  Path(__file__).resolve(), Path(select_training_features.__code__.co_filename).resolve()]
        self.source_fingerprints = {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in inputs}
        self.source_fingerprints.update({
            str((MANUAL_CACHE / n).relative_to(PROJECT_ROOT)): digest
            for n, digest in HOLDOUT_HASHES.items()
        })
        self.tsfresh_versions = {
            name: version(name) for name in ('tsfresh', 'numpy', 'scipy', 'pandas', 'statsmodels')
        }
        self.full_frames = {}
        self._blocks = lru_cache(maxsize=20)(self._prepare_block)

    def _cohort(self, fold):
        if fold is None:
            return self.membership, self.membership.split.eq('train')
        if fold not in self.train.fold.unique():
            raise ValueError(f'Unknown frozen fold: {fold}')
        return self.train, self.train.fold.ne(fold)

    def _fold_tsfresh(self, fold, fit_mask):
        path = self.cache_dir / f'tsfresh_fold_{fold}'
        expected = {
            'cohort_sha256': hashlib.sha256(self.train.to_csv(index=False).encode()).hexdigest(),
            'fit_rows': int(fit_mask.sum()),
            'fold': int(fold),
            'source': self.source_fingerprints,
            'versions': self.tsfresh_versions,
        }
        if not path.exists():
            if self.raw is None:
                self.raw = raw_tsfresh(self.membership, self.train, self.tsfresh_manifest)
            selected, ranges = select_training_features(
                self.raw, self.train.label, self.train.index[fit_mask], return_ranges=True,
            )
            ranges = {key: {name: float(value) for name, value in values.items()}
                      for key, values in zip(('max', 'min', 'median'), ranges, strict=True)}
            with TemporaryDirectory(dir=self.cache_dir) as temp:
                stage = Path(temp) / path.name
                stage.mkdir()
                array = stage / 'values.npy'
                np.save(array, selected.to_numpy(dtype=np.float64), allow_pickle=False)
                write_json(stage / 'manifest.json', {
                    **expected, 'columns': selected.columns.tolist(), 'imputation': ranges,
                    'values_sha256': sha256_file(array),
                })
                stage.rename(path)
            if all((self.cache_dir / f'tsfresh_fold_{i}').is_dir() for i in self.train.fold.unique()):
                self.raw = None
        manifest = read_json(path / 'manifest.json')
        if any(manifest[key] != value for key, value in expected.items()):
            raise ValueError(f'Tsfresh cache provenance mismatch: {path}')
        array = path / 'values.npy'
        require_hash(array, manifest['values_sha256'])
        data = np.load(array, mmap_mode='r', allow_pickle=False)
        if data.shape != (len(self.train), len(manifest['columns'])) or data.dtype != np.float64:
            raise ValueError(f'Tsfresh cache shape or dtype mismatch: {path}')
        return data, list(manifest['columns']), {
            'method': 'archived raw 777 candidates; imputation and selection refitted in the fold',
            'cache_manifest': str(path / 'manifest.json'),
            'cache_manifest_sha256': sha256_file(path / 'manifest.json'),
        }

    def _prepare_block(self, fold, name):
        cohort, fit = self._cohort(fold)
        if name == 'tsfresh' and fold is not None:
            array, cols, provenance = self._fold_tsfresh(fold, fit)
        elif name == 'tsfresh':
            if name not in self.full_frames:
                self.full_frames[name] = load_tsfresh_holdout(self.membership, self.tsfresh_manifest)
            frame = self.full_frames[name]
            array, cols = frame.to_numpy(dtype=np.float64), list(frame.columns)
            imputation_path = TSFRESH / 'imputation.csv'
            provenance = {
                'method': 'archived 708 features selected and imputed on frozen training partition only',
                'feature_table_sha256': self.tsfresh_manifest['features_sha256'],
                'imputation_path': str(imputation_path),
                'imputation_sha256': sha256_file(imputation_path),
            }
        else:
            if name == 'manual' and fold is None and name not in self.full_frames:
                self.full_frames[name] = load_manual_holdout(self.membership, self.frames[name])
            frame = self.full_frames[name] if name == 'manual' and fold is None else self.frames[name]
            frame = frame.loc[cohort.index]
            array = frame.to_numpy(dtype=np.float64, copy=True)
            array[~np.isfinite(array)] = np.nan
            imputer = SimpleImputer(strategy='median', keep_empty_features=True)
            imputer.fit(array[fit.to_numpy()])
            array = imputer.transform(array)
            cols = list(frame.columns)
            provenance = {'method': 'fit-row median; entirely empty fit columns subsequently removed',
                          'medians': {col: float(value) for col, value in zip(cols, imputer.statistics_, strict=True)}}
        prefixed = [f'{name}::{col}' for col in cols]
        signatures = training_column_signatures(array, fit.to_numpy())
        return array, prefixed, signatures, provenance

    def _assemble(self, fold, names):
        names = tuple(names)
        if (not names or len(set(names)) != len(names) or not set(names) <= set(INPUT_SETS)
                or {'physical_v1', 'physical_v2'} <= set(names)):
            raise ValueError(f'Invalid feature combination: {names}')
        cohort, fit = self._cohort(fold)
        fit_mask = fit.to_numpy()
        seen, constants, duplicates, kept, provenance = {}, [], {}, [], {}
        blocks = []
        for name in INPUT_SETS:
            if name not in names:
                continue
            array, columns, signatures, provenance[name] = self._blocks(fold, name)
            keep = []
            for i, (col, signature) in enumerate(zip(columns, signatures, strict=True)):
                if signature is None:
                    constants.append(col)
                    continue
                peers = seen.setdefault(signature, [])
                original = next((old_col for old, j, old_col in peers
                                 if np.array_equal(array[fit_mask, i], old[fit_mask, j])), None)
                if original is not None:
                    duplicates[col] = original
                else:
                    peers.append((array, i, col))
                    kept.append(col)
                    keep.append(i)
            if keep:
                blocks.append(array if len(keep) == len(columns) else array[:, keep])
        if not blocks:
            raise ValueError('All candidate feature columns are constant on the fit rows')
        X = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
        state = {'feature_sets': list(provenance), 'retained_features': kept, 'removed_constants': constants,
                 'removed_duplicates': duplicates, 'preprocessing_by_set': provenance}
        return X, state, cohort, fit_mask

    def fold(self, fold, names):
        X, state, _, fit = self._assemble(fold, names)
        return np.ascontiguousarray(X[fit]), np.ascontiguousarray(X[~fit]), state

    def holdout(self, names):
        X, state, cohort, fit = self._assemble(None, names)
        return (
            np.ascontiguousarray(X[fit]),
            np.ascontiguousarray(X[cohort.split.eq('validation')]),
            np.ascontiguousarray(X[cohort.split.eq('test')]),
            state,
        )
