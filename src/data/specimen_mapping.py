"""Coordinate-only specimen candidates: explicit post-catalog metadata pipeline.

No labels, features, splits or historical artifacts are read or modified.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from src.data.reference_measurements import REFERENCE_BOUNDS, reference_mask
from src.data.split_manifest import sha256_file
from src.experiments.provenance import capture_git_state
from src.paths import PROJECT_ROOT

EPS = 1.5  # includes unit-grid diagonals; excludes gaps of two coordinate units
MIN_SAMPLES = 4
MIN_CLUSTER_POINTS = 100
SMALL_FRACTION = 0.5  # review clusters below half the median cluster size
SUMMARY_COLUMNS = ['source_dataset', 'specimen_id', 'n_points', 'x_min', 'x_max',
                   'y_min', 'y_max', 'review_required']


def identify_specimens(catalog):
    """Return sorted mapping and bounds; ambiguous border/noise points stay null.

    Canonical coordinate order also makes DBSCAN border handling deterministic.
    IDs rank clusters by (minimum y, minimum x), separately within each source.
    Small spatial regions get candidate IDs but explicitly require manual review.
    """
    required = {'instance_id', 'source_dataset', 'x', 'y'}
    if not required <= set(catalog):
        raise ValueError('Missing coordinate metadata')
    df = catalog[list(sorted(required))].copy()
    if df.empty or df.isna().any().any() or not df.instance_id.is_unique:
        raise ValueError('Nonempty metadata with unique non-null instance_id required')
    if not np.isfinite(df[['x', 'y']].to_numpy(dtype=float)).all():
        raise ValueError('Finite coordinates required')
    if df.duplicated(['source_dataset', 'x', 'y']).any():
        raise ValueError('Duplicate source coordinates')
    rows, summaries = [], []
    for source, group in df.groupby('source_dataset', sort=True):
        group = group.sort_values(['x', 'y']).reset_index(drop=True)
        xy = group[['x', 'y']].to_numpy(dtype=float)
        model = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(xy)
        labels = model.labels_.copy()
        # Border points touching cores from multiple clusters have no unique owner.
        cores = model.core_sample_indices_
        ambiguous = np.zeros(len(group), dtype=bool)
        if len(cores):
            neighborhoods = cKDTree(xy[cores]).query_ball_point(xy, EPS)
            ambiguous = np.array([len(set(labels[cores[n]])) > 1 for n in neighborhoods])
        keys = sorted(set(labels) - {-1}, key=lambda k: (
            xy[labels == k, 1].min(), xy[labels == k, 0].min()))
        sizes = {k: int((labels == k).sum()) for k in keys}
        median = np.median(list(sizes.values())) if sizes else 0
        for i, k in enumerate(keys, 1):
            selected = (labels == k) & ~ambiguous
            if sizes[k] < MIN_CLUSTER_POINTS:
                continue
            sid = f'{source}_pk_{i:02d}'
            small = sizes[k] < SMALL_FRACTION * median
            a = xy[selected]
            if len(a):
                summaries.append([source, sid, len(a), a[:, 0].min(), a[:, 0].max(),
                                  a[:, 1].min(), a[:, 1].max(), bool(small)])
            for idx in np.flatnonzero(selected):
                rows.append([group.instance_id.iloc[idx], sid, source,
                             'review_small_region' if small else 'spatial_candidate'])
        assigned = {row[0] for row in rows}
        for idx, row in group.iterrows():
            if row.instance_id not in assigned:
                status = 'ambiguous_border' if ambiguous[idx] else 'noise_or_small_cluster'
                rows.append([row.instance_id, None, source, status])
    mapping = pd.DataFrame(rows, columns=['instance_id', 'specimen_id', 'source_dataset',
                                          'assignment_status']).sort_values('instance_id').reset_index(drop=True)
    reference = reference_mask(mapping)
    mapping['measurement_role'] = np.where(reference, 'reference', 'specimen')
    mapping['training_eligible'] = mapping.specimen_id.notna() & ~reference
    mapping.loc[reference, 'assignment_status'] = 'confirmed_reference'
    summary = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    reference_ids = set(mapping.loc[reference, 'specimen_id'])
    summary['measurement_role'] = np.where(summary.specimen_id.isin(reference_ids), 'reference', 'specimen')
    summary.loc[summary.specimen_id.isin(reference_ids), 'review_required'] = False
    return mapping, summary


def write_mapping(input_csv, output_dir):
    """Write a new sidecar directory, never overwrite an existing artifact."""
    input_csv, output_dir = Path(input_csv), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    before = sha256_file(input_csv)
    catalog = pd.read_csv(input_csv)
    mapping, summary = identify_specimens(catalog)
    if before != sha256_file(input_csv):
        raise ValueError('Catalog changed during mapping')
    state = capture_git_state(PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=False)
    mapping.to_csv(output_dir / 'specimen_mapping.csv', index=False)
    summary.to_csv(output_dir / 'specimen_summary.csv', index=False)
    catalog.loc[catalog.instance_id.isin(mapping.loc[mapping.training_eligible, 'instance_id'])].to_csv(
        output_dir / 'specimen_dataset.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points = catalog.merge(mapping[['instance_id', 'specimen_id']], on='instance_id', validate='one_to_one')
    for source, group in points.groupby('source_dataset', sort=True):
        fig, ax = plt.subplots(figsize=(10, 10))
        for i, (sid, g) in enumerate(group.groupby('specimen_id', dropna=False, sort=True)):
            color = plt.get_cmap('tab20')(i % 20) if pd.notna(sid) else 'black'
            ax.scatter(g.x, g.y, s=3, color=color)
            ax.text(g.x.mean(), g.y.mean(), f'{sid}\nn={len(g)}', ha='center', fontsize=7,
                    bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
        ax.set(xlabel='X (recorded units)', ylabel='Y (recorded units)',
               title=f'{source}: {group.specimen_id.nunique()} spatial candidates', aspect='equal')
        fig.tight_layout()
        fig.savefig(output_dir / f'{source}.png', dpi=150)
        plt.close(fig)
    manifest = {'format_version': 1, 'input_sha256': before,
                'coordinate_metadata_sha256': __import__('hashlib').sha256(
                    catalog[['instance_id', 'source_dataset', 'x', 'y']].sort_values('instance_id')
                    .to_csv(index=False).encode()).hexdigest(),
                'parameters': {'eps': EPS, 'min_samples': MIN_SAMPLES,
                               'confirmed_reference_bounds': REFERENCE_BOUNDS,
                               'min_cluster_points': MIN_CLUSTER_POINTS, 'small_fraction': SMALL_FRACTION},
                'ordering': 'source, cluster minimum y then minimum x; mapping sorted by instance_id',
                'source_sha256': sha256_file(Path(__file__)),
                'dependency_lock_sha256': sha256_file(PROJECT_ROOT / 'requirements-lock.txt'),
                'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib']},
                'git_state': state,
                'output_sha256': {p.name: sha256_file(p) for p in sorted(output_dir.iterdir())},
                'unassigned': int(mapping.specimen_id.isna().sum()),
                'interpretation': 'Upper isolated regions confirmed as references by dataset owner; excluded from training. Other regions remain spatial specimen candidates.'}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return mapping, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New metadata directory only')
    args = parser.parse_args()
    mapping, summary = write_mapping(args.input, args.output_dir)
    print(summary.to_string(index=False))
    print(f'Unassigned: {mapping.specimen_id.isna().sum()}')


if __name__ == '__main__':
    main()
