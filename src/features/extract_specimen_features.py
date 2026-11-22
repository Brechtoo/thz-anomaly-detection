"""Extract independent feature tables for the frozen specimen cohorts."""
import argparse
import importlib.metadata
import os
import platform
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.data.split_manifest import restore_splits, sha256_file
from src.experiments.provenance import capture_git_state
from src.features.loading import load_measurement
from src.features.specimen_inputs import (
    FOLDS,
    SPLITS,
    TSFRESH,
    load_cohorts,
    read_json,
    require_hash,
    tsfresh_manifest,
    write_json,
)
from src.paths import FEATURE_TABLES_ROOT, PROJECT_ROOT

DEFAULT_NAMES = {
    'tsfel': 'specimen_tsfel_v1',
    'physical': 'specimen_thz_physical_v1',
}
TSFEL = None


def uniform_trace(x, y):
    """Use the measured interval and preserve the trace extent and sample count."""
    if len(x) < 128 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid_trace')
    steps = np.diff(x)
    if (steps <= 0).any() or steps.max() > 1.5 * np.median(steps):
        raise ValueError('nonmonotonic_axis_or_gap')
    interval = float((x[-1] - x[0]) / (len(x) - 1))
    resampled = not np.allclose(steps, interval, rtol=1e-5, atol=0)
    if resampled:
        grid = np.linspace(x[0], x[-1], len(x))
        return grid, np.interp(grid, x, y), interval, resampled
    return x, y, interval, resampled


def init_worker(kind):
    if kind == 'tsfel':
        global TSFEL
        from src.features.specimen_tsfel import TsfelExtractor

        TSFEL = TsfelExtractor()


def extract_row(task):
    kind, path, digest = task
    require_hash(Path(path), digest)
    signal = load_measurement(path)
    detail = {'samples': len(signal), 'dx': None, 'resampled': False}
    try:
        x, y, dx, resampled = uniform_trace(signal.x.to_numpy(float), signal.y.to_numpy(float))
        detail.update(dx=dx, resampled=resampled)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', RuntimeWarning)
            if kind == 'tsfel':
                record, errors = TSFEL.extract(y, 1 / dx)
            else:
                from src.features.thz_physical import extract_physical

                record, errors = extract_physical(x, y), []
        return record, detail, errors, [str(w.message) for w in caught]
    except ValueError as exc:
        return {}, detail, [str(exc)], []


def feature_config(kind):
    if kind == 'tsfel':
        from src.features.specimen_tsfel import TsfelExtractor

        extractor = TsfelExtractor()
        return extractor.columns, {'calculators': extractor.config, 'omitted': extractor.omitted}
    from src.features.thz_physical import PHYSICAL_COLUMNS, PHYSICAL_CONFIG

    return PHYSICAL_COLUMNS, PHYSICAL_CONFIG


def extract_dataset(kind, output, workers):
    manifest_path = output.with_suffix('.json')
    diagnostics_path = output.with_suffix('.diagnostics.csv')
    if any(path.exists() for path in (output, manifest_path, diagnostics_path)):
        raise FileExistsError(output)
    if output.suffix != '.parquet' or workers < 1:
        raise ValueError('A .parquet output and at least one worker are required')

    started = time.perf_counter()
    membership, train = load_cohorts()
    tsfresh_manifest()  # Check the archived waveform hashes and frozen split association.
    paths = pd.read_csv(TSFRESH / 'raw_inputs.csv')
    if (not paths.instance_id.equals(membership.instance_id)
            or paths.sha256.tolist() != read_json(TSFRESH / 'extraction_inputs.json')['raw_hashes']):
        raise ValueError('Archived waveform IDs or hashes differ')
    columns, settings = feature_config(kind)
    expected = set(columns)
    values = np.full((len(paths), len(columns)), np.nan)
    summaries = []
    errors, warning_counts = Counter(), Counter()
    extraction_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(kind,)) as pool:
        tasks = ((kind, row.file_path, row.sha256) for row in paths.itertuples(index=False))
        for i, (features, detail, row_errors, row_warnings) in enumerate(pool.map(extract_row, tasks, chunksize=32)):
            if features:
                if set(features) != expected:
                    raise ValueError('Measurement feature schema differs from configured schema')
                values[i] = [features[name] for name in columns]
            errors.update(row_errors)
            warning_counts.update(row_warnings)
            summaries.append(dict(instance_id=paths.instance_id.iat[i], status='ok' if features else 'invalid',
                                  errors=' | '.join(row_errors), **detail))
            if (i + 1) % 5000 == 0:
                print(f'{kind}: {i + 1}/{len(paths)} measurements', flush=True)
    extraction_seconds = time.perf_counter() - extraction_start

    values[~np.isfinite(values)] = np.nan
    # An uninformative column everywhere cannot contribute to any partition.
    # Preserve constant finite values with variable missingness for fold-only handling.
    removed = {}
    for j, name in enumerate(columns):
        col = values[:, j]
        if np.isnan(col).all():
            removed[name] = 'undefined_everywhere'
        elif np.isfinite(col).all() and (col == col[0]).all():
            removed[name] = 'exact_constant_everywhere'
    df = pd.DataFrame(values, columns=columns).drop(columns=list(removed))
    if df.empty:
        raise ValueError('No usable feature columns after extraction')
    diagnostic = pd.DataFrame(summaries)
    diagnostic['missing_features'] = df.isna().sum(axis=1)
    restore_splits(df, membership.label, membership.instance_id, SPLITS)
    metadata = membership.assign(cv_fold=membership.specimen_id.map(
        train.drop_duplicates('specimen_id').set_index('specimen_id').fold,
    ).astype('Int64'))
    frame = pd.concat([metadata, df.add_prefix(f'{kind}__')], axis=1)
    details = {
        'kind': kind, 'rows': len(df), 'computed_features': len(columns),
        'retained_features': frame.columns[len(metadata.columns):].tolist(),
        'metadata_columns': metadata.columns.tolist(), 'configuration': settings,
        'removed_columns': removed, 'missing_by_feature': df.isna().sum().to_dict(),
        'complete_rows': int(df.notna().all(axis=1).sum()),
        'structurally_invalid_rows': int(diagnostic.status.eq('invalid').sum()),
        'rows_without_features': int(df.isna().all(axis=1).sum()),
        'errors': dict(errors), 'warnings': dict(warning_counts),
        'sampling': {
            'axis_unit': 'recorded X unit; physical time calibration undocumented',
            'dx_min': float(diagnostic.dx.min()), 'dx_max': float(diagnostic.dx.max()),
            'resampled_rows': int(diagnostic.resampled.sum()),
            'fs_rule': '(sample_count - 1) / (X_last - X_first)',
        },
        'imputation': 'none; invalid values preserved as null, except columns uninformative on all traces',
        'workers': workers, 'extraction_seconds': extraction_seconds,
        'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'git': capture_git_state(PROJECT_ROOT),
    }
    sources = {
        SPLITS / 'manifest.json', FOLDS / 'manifest.json', TSFRESH / 'manifest.json',
        TSFRESH / 'raw_inputs.csv', Path(__file__).resolve(),
        Path(__file__).with_name('specimen_tsfel.py' if kind == 'tsfel' else 'thz_physical.py'),
        Path(__file__).with_name('experimental_anchors.py'), Path(__file__).with_name('loading.py'),
    }
    details['inputs_sha256'] = {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in sorted(sources)}
    output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        scratch = Path(scratch)
        parquet = scratch / output.name
        diagnostic_csv = scratch / diagnostics_path.name
        manifest = scratch / manifest_path.name
        frame.to_parquet(parquet, index=False, compression='zstd')
        pd.testing.assert_frame_equal(
            frame, pd.read_parquet(parquet), check_exact=True,
        )
        diagnostic.to_csv(diagnostic_csv, index=False)
        details.update(parquet_sha256=sha256_file(parquet), diagnostics_sha256=sha256_file(diagnostic_csv),
                       total_seconds=time.perf_counter() - started)
        write_json(manifest, details)
        for source, target in ((parquet, output), (diagnostic_csv, diagnostics_path), (manifest, manifest_path)):
            os.link(source, target)  # Publish without overwriting a pre-existing file.
    print(f'{kind}: wrote {len(frame)} measurements, {df.shape[1]} features to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=DEFAULT_NAMES)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    extract_dataset(args.kind, args.output or FEATURE_TABLES_ROOT / f'{DEFAULT_NAMES[args.kind]}.parquet', args.workers)


if __name__ == '__main__':
    main()
