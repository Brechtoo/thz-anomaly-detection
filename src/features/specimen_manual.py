"""Reuse audited manual features, then fit median imputation on frozen training IDs.

The archived independent rebuild supplies evidence of numerical reproducibility;
current measurement hashes and extractor sources must still match that evidence.
No measurements are extracted and no source table or split is modified.
"""
import argparse
import ast
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.data.dataset import EXCLUDE_COLUMNS
from src.data.specimen_splits import load_membership
from src.data.split_manifest import restore_splits, sha256_file
from src.paths import PROJECT_ROOT


def validate_manual_schema(columns):
    """Reject metadata even if an input table happens to contain 52 columns."""
    metadata = set(EXCLUDE_COLUMNS) | {'specimen_id', 'measurement_role', 'training_eligible',
                                      'assignment_status', 'filename', 'z', 'split'}
    if len(columns) != 52 or len(set(columns)) != 52 or set(columns) & metadata:
        raise ValueError('Exactly 52 unique non-metadata feature columns required')


def verify_extractor_sources(manifest):
    """Allow formatting changes only after verifying the exact archived source.

    Reconstruct dirty historical sources from the recorded commit and patch in
    a temporary directory. Neither the working tree nor Git index is modified.
    AST comparison ignores comments/spacing, but rejects changed calculations.
    """
    names = ['src/features/' + name for name in ['loading.py', 'extract_all_features.py',
             'extract_baseline_features.py', 'extract_extension_block1_features.py']]
    current = {name: sha256_file(PROJECT_ROOT / name) for name in names}
    if all(current[name] == manifest['source_sha256'][name] for name in names):
        return {'current_sha256': current, 'comparison': 'byte_identical'}
    with tempfile.TemporaryDirectory(prefix='manual-feature-source-') as temporary:
        root = Path(temporary)
        state = manifest['git_state']
        for name in names:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(
                ['git', '-C', str(PROJECT_ROOT), 'show', state['git_commit'] + ':' + name]))
        patch = state.get('tracked_diff', '')
        if patch:
            patch_path = root / 'recorded.patch'
            patch_path.write_text(patch)
            subprocess.run(['git', 'apply', *['--include=' + name for name in names], str(patch_path)],
                           cwd=root, check=True, capture_output=True)
        archived = {}
        for name in names:
            archived[name] = sha256_file(root / name)
            if archived[name] != manifest['source_sha256'][name]:
                raise ValueError(f'Cannot reconstruct archived source: {name}')
            old = ast.dump(ast.parse((root / name).read_text()), include_attributes=False)
            new = ast.dump(ast.parse((PROJECT_ROOT / name).read_text()), include_attributes=False)
            if old != new:
                raise ValueError(f'Extractor calculations changed: {name}')
    return {'current_sha256': current, 'archived_sha256': archived, 'comparison': 'AST_identical',
            'archived_commit': state['git_commit'], 'patch_sha256': hashlib.sha256(patch.encode()).hexdigest()}


def verify_existing_features(table_path, catalog_path, evidence_dir, columns):
    """Verify all rows against the independent rebuild and current raw inputs."""
    evidence_dir = Path(evidence_dir)
    manifest_path = evidence_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    rebuilt_path = evidence_dir / 'feature_table.csv'
    raw_path = evidence_dir / 'raw_inputs.csv'
    if (sha256_file(rebuilt_path) != manifest['output_csv_sha256']
            or sha256_file(raw_path) != manifest['raw_inputs_sha256']
            or sha256_file(catalog_path) != manifest['input_csv_sha256']):
        raise ValueError('Rebuild evidence checksum mismatch')
    if manifest['feature_columns'] != columns:
        raise ValueError('Manual feature schema differs from independent rebuild')
    sources = verify_extractor_sources(manifest)
    table = pd.read_csv(table_path)
    rebuilt = pd.read_csv(rebuilt_path)
    catalog = pd.read_csv(catalog_path)
    for frame in [table, rebuilt, catalog]:
        if frame.instance_id.isna().any() or not frame.instance_id.is_unique:
            raise ValueError('Unique non-null measurement IDs required')
    if set(table.instance_id) != set(catalog.instance_id) or set(table.instance_id) != set(rebuilt.instance_id):
        raise ValueError('Incomplete feature table')
    metadata = EXCLUDE_COLUMNS
    left = table.sort_values('instance_id').reset_index(drop=True)
    right = rebuilt.sort_values('instance_id').reset_index(drop=True)
    original = catalog.sort_values('instance_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(left[metadata], right[metadata], check_dtype=False)
    pd.testing.assert_frame_equal(left[metadata], original[metadata], check_dtype=False)
    if set(table) != set(columns) | set(metadata):
        raise ValueError('Unexpected feature table columns')
    a, b = left[columns].to_numpy(), right[columns].to_numpy()
    if not np.array_equal(np.isfinite(a), np.isfinite(b)) or not np.array_equal(np.isnan(a), np.isnan(b)):
        raise ValueError('Feature finiteness masks differ')
    if not np.isclose(a, b, rtol=1e-11, atol=1e-13, equal_nan=True).all():
        raise ValueError('Numerical feature reproducibility check failed')
    raw = pd.read_csv(raw_path)
    if not raw.instance_id.is_unique or set(raw.instance_id) != set(table.instance_id):
        raise ValueError('Raw evidence coverage differs')
    print(f'Checking {len(raw)} original measurement checksums ...', flush=True)
    for row in raw.itertuples(index=False):
        if sha256_file(row.file_path) != row.sha256:
            raise ValueError(f'Raw measurement changed: {row.instance_id}')
    return {'rows': len(table), 'features': len(columns), 'new_extraction': False,
            'table_sha256': sha256_file(table_path), 'rebuild_manifest_sha256': sha256_file(manifest_path),
            'rebuilt_table_sha256': sha256_file(rebuilt_path), 'raw_inputs_sha256': sha256_file(raw_path),
            'verified_raw_files': len(raw), 'extractor_sha256': sources,
            'rtol': 1e-11, 'atol': 1e-13, 'max_absolute_difference': float(np.nanmax(np.abs(a-b))),
            'interpretation': 'Numerical agreement with independent raw-data rebuild; not byte equality.'}


def prepare_manual_features(table_path, catalog_path, split_dir, schema_path, evidence_dir, output_dir):
    """Reuse the original 52 values and labels; impute using training medians only."""
    output_dir, split_dir = Path(output_dir), Path(split_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    schema = json.loads(Path(schema_path).read_text())
    columns = schema['features']
    validate_manual_schema(columns)
    if sha256_file(table_path) != schema['source_table_sha256']:
        raise ValueError('Manual source table checksum mismatch')
    verified = verify_existing_features(table_path, catalog_path, evidence_dir, columns)
    membership = load_membership(split_dir)
    source = pd.read_csv(table_path).set_index('instance_id')
    data = source.loc[membership.instance_id].reset_index()
    if not np.array_equal(data.label.to_numpy(), membership.label.to_numpy()):
        raise ValueError('Manual labels differ from frozen splits')
    X = data[columns].replace([np.inf, -np.inf], np.nan)
    train, *_ = restore_splits(X, data.label, data.instance_id, split_dir)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(train)
    transformed = pd.DataFrame(imputer.transform(X), columns=columns, index=data.index)
    if not np.isfinite(transformed.to_numpy()).all():
        raise ValueError('Manual imputation left non-finite features')
    output_dir.mkdir(parents=True, exist_ok=False)
    table = pd.concat([data[['instance_id', 'label']], transformed], axis=1)
    table.to_csv(output_dir / 'features.csv', index=False)
    pd.DataFrame({'feature': columns, 'training_median': imputer.statistics_}).to_csv(output_dir / 'imputation.csv', index=False)
    joblib.dump(imputer, output_dir / 'imputer.joblib')
    (output_dir / 'verification.json').write_text(json.dumps(verified, indent=2) + '\n')
    (output_dir / 'feature_schema.json').write_text(json.dumps(schema, indent=2) + '\n')
    raw = pd.read_csv(Path(evidence_dir) / 'raw_inputs.csv').set_index('instance_id')
    raw.loc[membership.instance_id].reset_index().to_csv(output_dir / 'raw_inputs.csv', index=False)
    inputs = {'split_sha256': sha256_file(split_dir / 'manifest.json'),
              'source_table_sha256': sha256_file(table_path),
              'feature_schema_sha256': sha256_file(output_dir / 'feature_schema.json'),
              'preprocessing': 'SimpleImputer median, keep_empty_features=True; fit only frozen training IDs',
              'training_rows': len(train), 'source_sha256': sha256_file(__file__)}
    (output_dir / 'extraction_inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
    manifest = {'feature_set': 'manual52', 'input_sha256': sha256_file(output_dir / 'extraction_inputs.json'),
                'features_sha256': sha256_file(output_dir / 'features.csv'), 'selected_features': columns,
                'training_only': True, 'reused_existing_values': True,
                'imputer_sha256': sha256_file(output_dir / 'imputer.joblib')}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Reused {len(columns)} features for {len(table)} eligible points; no extraction needed', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['table', 'catalog', 'split-dir', 'schema', 'evidence-dir', 'output-dir']:
        p.add_argument('--' + name, required=True, type=Path)
    args = p.parse_args()
    prepare_manual_features(args.table, args.catalog, args.split_dir, args.schema, args.evidence_dir, args.output_dir)


if __name__ == '__main__':
    main()
