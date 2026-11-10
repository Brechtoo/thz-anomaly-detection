"""Audit cross-table numerical equality; formula identities are listed explicitly.

Run with --baseline, --tsfresh and a new --output JSON path. No features change.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import EXCLUDE_COLUMNS
from src.data.split_manifest import sha256_file

# Verified against the local extraction code and tsfresh 0.21.2 calculators.
IDENTICAL_FORMULAS = {
    'y_max': 'y__maximum', 'y_min': 'y__minimum', 'y_mean': 'y__mean',
    'y_median': 'y__median', 'y_std': 'y__standard_deviation',
    'y_energy': 'y__abs_energy',
}


def audit_overlap(baseline, tsfresh, output):
    baseline, tsfresh, output = map(Path, (baseline, tsfresh, output))
    if output.exists():
        raise FileExistsError(output)
    hashes = {str(p): sha256_file(p) for p in [baseline, tsfresh]}
    base = pd.read_csv(baseline).set_index('instance_id', verify_integrity=True)
    columns = [c for c in base if c not in EXCLUDE_COLUMNS]
    header = pd.read_csv(tsfresh, nrows=0)
    tscols = [c for c in header if c not in EXCLUDE_COLUMNS]
    # Numerical candidates on 32 rows; every survivor is checked on ALL rows.
    sample = pd.read_csv(tsfresh, nrows=32).set_index('instance_id', verify_integrity=True)
    aligned = base.loc[sample.index]
    pairs = [(b, t) for b in columns for t in tscols
             if np.allclose(aligned[b], sample[t], rtol=1e-10, atol=1e-12, equal_nan=False)]
    checks = {p: {'equal': True, 'max_absolute_difference': 0.0} for p in pairs}
    seen = set()
    count = 0
    for chunk in pd.read_csv(tsfresh, chunksize=2048,
                             usecols=['instance_id'] + sorted({t for _, t in pairs})):
        chunk = chunk.set_index('instance_id', verify_integrity=True)
        if seen.intersection(chunk.index):
            raise ValueError('Duplicate tsfresh IDs')
        seen.update(chunk.index)
        current = base.loc[chunk.index]
        count += len(chunk)
        for b, t in pairs:
            left, right = current[b].to_numpy(), chunk[t].to_numpy()
            check = checks[(b, t)]
            check['equal'] &= bool(np.allclose(left, right, rtol=1e-10, atol=1e-12))
            check['max_absolute_difference'] = max(check['max_absolute_difference'],
                                                    float(np.max(np.abs(left - right))))
    if seen != set(base.index):
        raise ValueError('ID coverage differs')
    if any(sha256_file(p) != h for p, h in hashes.items()):
        raise ValueError('Inputs changed during audit')
    result = {'input_sha256': hashes, 'rows': count, 'baseline_features': len(columns),
              'tsfresh_features': len(tscols), 'rtol': 1e-10, 'atol': 1e-12,
              'formula_identical_pairs': IDENTICAL_FORMULAS,
              'numerical_matches': [{'baseline': b, 'tsfresh': t, **check}
                                    for (b, t), check in checks.items() if check['equal']],
              'note': 'Numerical agreement alone does not prove formula identity. Energy uses sum of squares vs dot product; rounding differences allowed.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'tsfresh', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_overlap(args.baseline, args.tsfresh, args.output), indent=2))


if __name__ == '__main__':
    main()
