"""Derive a conservative, same-trace candidate variant without editing V1."""
import argparse
import importlib.metadata
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.data.split_manifest import sha256_file
from src.features import thz_physical as v1
from src.features.loading import load_measurement
from src.features.specimen_inputs import (
    TSFRESH,
    load_cohorts,
    read_json,
    require_hash,
    tsfresh_manifest,
    write_json,
)
from src.features.thz_echo_candidates import (
    candidates,
    correlation_curve,
    nonoverlapping_second,
    trace_windows,
)
from src.paths import FEATURE_TABLES_ROOT, PROJECT_ROOT

V1_PATH = FEATURE_TABLES_ROOT / 'specimen_thz_physical_v1.parquet'
V2_PATH = FEATURE_TABLES_ROOT / 'specimen_thz_physical_v2.parquet'
REPLACED = ('physical__ncc__secondary_to_primary', 'physical__ncc__peak_separation_over_reference')
ADDED = (
    'physical_v2__ncc__disjoint_secondary_to_primary',
    'physical_v2__ncc__disjoint_peak_separation_over_reference',
    'physical_v2__ncc__best_abs_rms_eligible',
    'physical_v2__ncc__rms_eligible_fraction',
)


def candidate_features(x, y):
    """Keep V1's gates and require a nonoverlapping window for the second candidate."""
    t, values, dx, lo, hi, start, end = trace_windows(x, y)
    score, rms = correlation_curve(values[lo:hi], values[start:end])
    peaks = candidates(score, rms, dx)
    second = nonoverlapping_second(peaks, hi - lo)
    eligible = rms > v1.MIN_RMS_FRACTION
    result = np.full(len(ADDED), np.nan)
    if second is not None:
        result[:2] = abs(score[second] / score[peaks[0]]), abs(second - peaks[0]) / (hi - lo)
    if eligible.any():
        result[2] = np.max(np.abs(score[eligible]))
    result[3] = eligible.mean()
    return result, len(peaks) * (hi - lo) / (end - start), (len(t), dx)


def extract_row(task):
    path, digest = task
    require_hash(Path(path), digest)
    wave = load_measurement(path)
    values, density, axis = candidate_features(wave.x.to_numpy(), wave.y.to_numpy())
    return values, density, (*axis, float(wave.x.iloc[0]), float(wave.x.iloc[-1]))


def derive(output, workers):
    manifest_path = output.with_suffix('.json')
    if output.suffix != '.parquet' or workers < 1:
        raise ValueError('Expected a .parquet path and a positive worker count')
    if output.exists() or manifest_path.exists():
        raise FileExistsError(output)
    if output.resolve() == V1_PATH:
        raise ValueError('V1 is read-only')
    started = time.perf_counter()
    v1_manifest_path = V1_PATH.with_suffix('.json')
    manifest = read_json(v1_manifest_path)
    protected = {
        V1_PATH: manifest['parquet_sha256'],
        v1_manifest_path: sha256_file(v1_manifest_path),
        V1_PATH.with_suffix('.diagnostics.csv'): manifest['diagnostics_sha256'],
        PROJECT_ROOT / 'src/features/thz_physical.py': manifest['inputs_sha256']['src/features/thz_physical.py'],
    }
    for path, digest in protected.items():
        require_hash(path, digest)
    membership, train = load_cohorts()
    table = pd.read_parquet(V1_PATH)
    pd.testing.assert_frame_equal(table[membership.columns], membership)
    np.testing.assert_array_equal(table.cv_fold.dropna(), train.fold)
    assert table.cv_fold.notna().equals(table.split.eq('train'))
    tsfresh_manifest()
    raw_path = TSFRESH / 'raw_inputs.csv'
    raw = pd.read_csv(raw_path)
    if (not raw.instance_id.equals(table.instance_id)
            or raw.sha256.tolist() != read_json(TSFRESH / 'extraction_inputs.json')['raw_hashes']):
        raise ValueError('Unexpected archive membership or raw hashes')
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = ((row.file_path, row.sha256) for row in raw.itertuples(index=False))
        for i, record in enumerate(pool.map(extract_row, tasks, chunksize=32), start=1):
            records.append(record)
            if i % 10000 == 0:
                print(f'V2: {i}/{len(raw)}', flush=True)
    matrix = pd.DataFrame(np.vstack([row[0] for row in records]), columns=ADDED)
    density = np.array([row[1] for row in records])
    np.testing.assert_array_equal(density, table['physical__ncc__match_count_per_reference_length'])
    if (matrix[ADDED[0]].notna() & table[REPLACED[0]].isna()).any() or (matrix[ADDED[1]].dropna() < 1).any():
        raise ValueError('Invalid disjoint candidates')
    # Full-cohort checks describe the export; no sample was dropped and no statistics feed the extractor.
    for name, series in matrix.items():
        if np.isinf(series).any() or series.nunique() <= 1:
            raise ValueError(f'Uninformative or invalid derived feature: {name}')
    result = pd.concat([table.drop(columns=list(REPLACED)), matrix], axis=1)
    axes = pd.DataFrame([row[2] for row in records], columns=[
        'samples', 'dx', 'raw_start', 'raw_end',
    ])
    settings = {
        'version': 'specimen_thz_physical_v2',
        'configuration': {'detector': v1.PHYSICAL_CONFIG,
                          'secondary': 'highest-ranked V1 candidate whose full window is disjoint from the primary',
                          'best_abs': 'max absolute local Pearson score before peak selection, over RMS-eligible windows',
                          'eligibility': 'fraction of positions with local RMS above the V1 gate'},
        'retained_features': [name for name in result if name.startswith(('physical__', 'physical_v2__'))],
        'rows': len(result), 'replaced': list(REPLACED),
        'metadata_columns': manifest['metadata_columns'],
        'raw_axis_min_max': {name: [float(axes[name].min()), float(axes[name].max())]
                             for name in ('samples', 'dx', 'raw_start', 'raw_end')},
        'missing_v1_second': int(table[REPLACED[0]].isna().sum()),
        'missing_v2': matrix.isna().sum().to_dict(),
        'complete_rows_v1': int(table[manifest['retained_features']].notna().all(axis=1).sum()),
        'complete_rows_v2': int(result.drop(columns=manifest['metadata_columns']).notna().all(axis=1).sum()),
        'v2_value_summary': matrix.describe().to_dict(),
        'raw_hash_catalog_sha256': sha256_file(raw_path), 'workers': workers,
        'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'pyarrow')},
        'input_sha256': {str(path.relative_to(PROJECT_ROOT)): digest for path, digest in protected.items()},
    }
    settings['input_sha256'].update({
        f'src/features/{name}': sha256_file(Path(__file__).with_name(name))
        for name in ('thz_physical_v2.py', 'thz_echo_candidates.py', 'experimental_anchors.py', 'loading.py')
    })
    for path, digest in protected.items():
        require_hash(path, digest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        parquet = Path(scratch) / output.name
        result.to_parquet(parquet, compression='zstd', index=False)
        pd.testing.assert_frame_equal(result, pd.read_parquet(parquet), check_exact=True)
        settings.update(parquet_sha256=sha256_file(parquet), seconds=time.perf_counter()-started)
        staged_manifest = Path(scratch) / manifest_path.name
        write_json(staged_manifest, settings)
        output.hardlink_to(parquet)
        manifest_path.hardlink_to(staged_manifest)
    return settings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=V2_PATH)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    derive(args.output, args.workers)
