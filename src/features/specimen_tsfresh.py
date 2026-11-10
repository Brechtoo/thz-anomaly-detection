"""Fresh raw tsfresh cache and train-only selection for a frozen specimen split.

Batches are checkpointed to avoid losing a long extraction. Resumption verifies
catalog, split manifest, extractor source and every raw-file checksum first.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tsfresh.utilities.dataframe_functions import get_range_values_per_column

from src.data.split_manifest import restore_splits, sha256_file
from src.features.extract_tsfresh_features import (
    extract_measurement,
    select_training_features,
)
from src.paths import PROJECT_ROOT


def build_specimen_features(split_dir, output_dir, workers=8):
    split_dir, output_dir = Path(split_dir), Path(output_dir)
    data = pd.read_csv(split_dir / 'membership.csv').sort_values('instance_id').reset_index(drop=True)
    catalog_path = PROJECT_ROOT / 'data/measurements/combined_dataset.csv'
    catalog = pd.read_csv(catalog_path).set_index('instance_id')
    paths = catalog.loc[data.instance_id, 'file_path'].tolist()
    raw_hashes = [sha256_file(p) for p in paths]
    sources = [Path(__file__), PROJECT_ROOT / 'src/features/extract_tsfresh_features.py',
               PROJECT_ROOT / 'src/features/loading.py']
    inputs = {'catalog_sha256': sha256_file(catalog_path),
              'split_sha256': sha256_file(split_dir / 'manifest.json'),
              'source_sha256': {str(p): sha256_file(p) for p in sources},
              'raw_hashes': raw_hashes, 'preset': 'efficient', 'batch_size': 512}
    output_dir.mkdir(parents=True, exist_ok=True)
    config = output_dir / 'extraction_inputs.json'
    if config.exists():
        if json.loads(config.read_text()) != inputs:
            raise ValueError('Extraction inputs changed; new output directory required')
    else:
        config.write_text(json.dumps(inputs, indent=2) + '\n')
        pd.DataFrame({'instance_id': data.instance_id, 'file_path': paths, 'sha256': raw_hashes}).to_csv(
            output_dir / 'raw_inputs.csv', index=False)
    cache = output_dir / 'raw_batches'
    cache.mkdir(exist_ok=True)
    arrays = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(data), 512):
            target = cache / f'{start:06d}.npz'
            if not target.exists():
                records = list(pool.map(extract_measurement,
                                       [(p, 'efficient') for p in paths[start:start+512]], chunksize=16))
                names = list(records[0])
                values = np.array([[record[n] for n in names] for record in records])
                temporary = target.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, values=values, names=names)
                temporary.rename(target)
            with np.load(target) as saved:
                names = saved['names'].tolist()
                arrays.append(saved['values'])
            print(f'Raw features {min(start+512, len(data))}/{len(data)}', flush=True)
    if raw_hashes != [sha256_file(p) for p in paths]:
        raise ValueError('Raw inputs changed during extraction')
    features = pd.DataFrame(np.concatenate(arrays), columns=names)
    train = data.index[data.split.eq('train')]
    ranges = get_range_values_per_column(features.loc[train].copy())
    selected = select_training_features(features, data.label, train, n_jobs=workers)
    result = pd.concat([data[['instance_id', 'label']], selected], axis=1)
    target = output_dir / 'features.csv'
    if target.exists():
        raise FileExistsError(target)
    result.to_csv(target, index=False, mode='x')
    # Persist imputation state and all raw calculator columns for later inference.
    pd.DataFrame({'max': ranges[0], 'min': ranges[1], 'median': ranges[2]}).to_csv(output_dir / 'imputation.csv')
    manifest = {'input_sha256': sha256_file(config), 'features_sha256': sha256_file(target),
                'raw_features': len(names), 'selected_features': selected.columns.tolist(),
                'training_only': True, 'raw_batches_sha256': {p.name: sha256_file(p) for p in sorted(cache.glob('*.npz'))}}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    restore_splits(selected, data.label, data.instance_id, split_dir)
    print(f'Selected {selected.shape[1]} of {len(names)} features', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    build_specimen_features(args.split_dir, args.output_dir, args.workers)


if __name__ == '__main__':
    main()
