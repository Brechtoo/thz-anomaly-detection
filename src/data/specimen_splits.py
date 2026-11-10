"""New grouped holdout manifests; candidate choice uses metadata only.

GroupShuffleSplit offers 7 test + 5 validation + 23 training groups, whereas
StratifiedGroupKFold's equal folds cannot directly express 65/15/20 with 35
unequal, mixed-label specimens. A fixed candidate budget balances counts without
consulting features or model performance. Canonical group/ID order fixes ties.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.reference_measurements import reference_mask
from src.data.split_manifest import SPLIT_NAMES, restore_splits, sha256_file

REFERENCES = {'messreihe_1_pk_15', 'messreihe_2_pk_15', 'messreihe_3_pk_08'}
RATIOS = np.array([.65, .15, .20])


def validate_membership(data):
    """Fail on partial IDs, group overlap, references or invalid split names."""
    required = ['instance_id', 'specimen_id', 'source_dataset', 'label', 'split']
    if data[required].isna().any().any() or not data.instance_id.is_unique:
        raise ValueError('Unique IDs and complete split metadata required')
    if set(data.split) != set(SPLIT_NAMES) or not data.label.isin(range(4)).all():
        raise ValueError('Invalid splits or labels')
    if (data.groupby('specimen_id').split.nunique() != 1).any():
        raise ValueError('Specimen overlap between splits')
    if data.specimen_id.isin(REFERENCES).any() or reference_mask(data).any():
        raise ValueError('References in split membership')


def load_membership(split_dir):
    """Verify sidecar hashes as well as the original v1 ID/label manifests."""
    directory = Path(split_dir)
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, expected in manifest['metadata_sha256'].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f'Split metadata checksum mismatch: {name}')
    data = pd.read_csv(directory / 'membership.csv')
    validate_membership(data)
    parts = restore_splits(data[['instance_id']], data.label, data.instance_id, directory)[:3]
    for name, part in zip(SPLIT_NAMES, parts):
        if not data.loc[part.index, 'split'].eq(name).all():
            raise ValueError('Membership disagrees with split ID files')
    return data


def eligible_catalog(catalog, mapping):
    """Join bijectively, exclude confirmed references, reject uncertain rows."""
    for frame in (catalog, mapping):
        if frame.instance_id.isna().any() or not frame.instance_id.is_unique:
            raise ValueError('Unique non-null IDs required')
    if set(catalog.instance_id) != set(mapping.instance_id):
        raise ValueError('Mapping must cover exactly the catalog IDs')
    result = catalog.merge(mapping[['instance_id', 'specimen_id']], on='instance_id', validate='one_to_one')
    excluded = result.specimen_id.isin(REFERENCES)
    if not excluded.equals(reference_mask(result)):
        raise ValueError('Reference ID and coordinate definitions disagree')
    result = result.loc[~excluded].sort_values('instance_id').reset_index(drop=True)
    if result.empty or result.specimen_id.isna().any() or not result.label.isin([0, 1, 2, 3]).all():
        raise ValueError('Complete specimen IDs and original labels required')
    if (result.groupby('specimen_id').source_dataset.nunique() != 1).any():
        raise ValueError('Specimen spans multiple sources')
    return result


def choose_groups(data, seed=42, candidates=4096):
    """Minimize mean squared allocation error for rows, labels and sources.

    Every split must contain all sources and all labels. Scores compare the
    fraction of each global class/source allocated to each split to its target.
    No model metrics or feature values enter candidate generation or scoring.
    """
    if candidates < 1:
        raise ValueError('Positive candidate budget required')
    groups = sorted(data.specimen_id.unique())
    labels = pd.crosstab(data.specimen_id, data.label).reindex(index=groups, columns=range(4), fill_value=0)
    sources = pd.crosstab(data.specimen_id, data.source_dataset).reindex(index=groups, fill_value=0)
    counts = np.column_stack([labels.sum(axis=1), labels, sources]).astype(float)
    if (counts.sum(axis=0) == 0).any():
        raise ValueError('All four labels required')
    best = None
    splitter = GroupShuffleSplit(n_splits=candidates, test_size=.20, random_state=seed)
    indices = np.arange(len(groups))
    for candidate, (remaining, test) in enumerate(splitter.split(indices, groups=groups)):
        inner = GroupShuffleSplit(n_splits=1, test_size=max(1, round(len(groups) * .15)),
                                  random_state=seed + candidate + 1)
        train_local, val_local = next(inner.split(remaining, groups=np.array(groups)[remaining]))
        parts = [remaining[train_local], remaining[val_local], test]
        allocated = np.array([counts[p].sum(axis=0) for p in parts])
        if (allocated[:, 1:] == 0).any():
            continue
        score = float(np.mean((allocated / counts.sum(axis=0) - RATIOS[:, None]) ** 2))
        if best is None or score < best[0]:
            best = score, candidate, parts
    if best is None:
        raise ValueError('No candidate covers every label and source in every split')
    score, candidate, parts = best
    assignments = {groups[i]: split for split, part in zip(SPLIT_NAMES, parts) for i in part}
    return assignments, {'seed': seed, 'candidates': candidates, 'selected_candidate': candidate,
                         'score': score, 'target_ratios': RATIOS.tolist(),
                         'objective': 'mean squared allocation fraction error: rows, four labels, sources'}


def write_specimen_splits(catalog_csv, mapping_csv, output_dir, seed=42, candidates=4096):
    """Keep the restore_splits v1 schema; store specimen metadata separately."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    hashes = {str(p): sha256_file(p) for p in [catalog_csv, mapping_csv]}
    data = eligible_catalog(pd.read_csv(catalog_csv), pd.read_csv(mapping_csv))
    assignment, config = choose_groups(data, seed, candidates)
    data['split'] = data.specimen_id.map(assignment)
    validate_membership(data)
    output_dir.mkdir(parents=True, exist_ok=False)
    files, rows = {}, []
    for name in SPLIT_NAMES:
        part = data.loc[data.split.eq(name)]
        path = output_dir / f'{name}.csv'
        part[['instance_id', 'label']].to_csv(path, index=False)
        files[path.name] = sha256_file(path)
        rows.append({'split': name, 'points': len(part), 'specimens': part.specimen_id.nunique(),
                     **{f'class_{k}': int(part.label.eq(k).sum()) for k in range(4)},
                     **{source: int(part.source_dataset.eq(source).sum()) for source in sorted(data.source_dataset.unique())}})
    data[['instance_id', 'specimen_id', 'source_dataset', 'label', 'split']].to_csv(output_dir / 'membership.csv', index=False)
    data.groupby(['split', 'source_dataset', 'specimen_id', 'label']).size().unstack(fill_value=0).to_csv(output_dir / 'specimens.csv')
    pd.DataFrame(rows).to_csv(output_dir / 'summary.csv', index=False)
    manifest = {'format_version': 1, 'files': files, 'group_key': 'specimen_id',
                'input_sha256': hashes, 'configuration': config, 'excluded_specimens': sorted(REFERENCES),
                'source_sha256': sha256_file(__file__),
                'metadata_sha256': {p.name: sha256_file(p) for p in output_dir.glob('*.csv')},
                'limitation': 'Physical identity across measurement series is unknown; spatial groups alone cannot exclude reuse.'}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    restore_splits(data[['instance_id']], data.label, data.instance_id, output_dir)
    if any(sha256_file(p) != h for p, h in hashes.items()):
        raise ValueError('Inputs changed')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--mapping', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    data = write_specimen_splits(args.catalog, args.mapping, args.output_dir)
    print(data.groupby('split').agg(points=('instance_id', 'size'), specimens=('specimen_id', 'nunique')))


if __name__ == '__main__':
    main()
