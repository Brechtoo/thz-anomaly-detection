"""Synthetic mixed-label specimens exercise grouping and saved v1 compatibility."""
import pandas as pd
import pytest

from src.data.specimen_splits import (
    choose_groups,
    eligible_catalog,
    load_membership,
    validate_membership,
    write_specimen_splits,
)


def synthetic():
    rows, mapping = [], []
    for source, size in [('a', 14), ('b', 14), ('c', 7)]:
        for group in range(size):
            for label in range(4):
                for point in range(1 + (group + label) % 3):
                    iid = f'{source}_{group}_{label}_{point}'
                    rows.append([iid, label, source, group, point])
                    mapping.append([iid, f'{source}_pk_{group:02d}'])
    return (pd.DataFrame(rows, columns=['instance_id', 'label', 'source_dataset', 'x', 'y']),
            pd.DataFrame(mapping, columns=['instance_id', 'specimen_id']))


def test_deterministic_grouped_assignment_and_complete_manifest(tmp_path):
    catalog, mapping = synthetic()
    data = eligible_catalog(catalog, mapping)
    assignments, config = choose_groups(data, candidates=32)
    shuffled, other = choose_groups(data.sample(frac=1, random_state=9), candidates=32)
    assert assignments == shuffled and config == other
    paths = tmp_path / 'catalog.csv', tmp_path / 'mapping.csv'
    catalog.to_csv(paths[0], index=False)
    mapping.to_csv(paths[1], index=False)
    before = [p.read_bytes() for p in paths]
    first = tmp_path / 'first'
    result = write_specimen_splits(*paths, first, candidates=32)
    validate_membership(result)
    assert result.groupby('split').specimen_id.nunique().to_dict() == {'train': 23, 'validation': 5, 'test': 7}
    assert set(result.instance_id) == set(catalog.instance_id)
    assert [p.read_bytes() for p in paths] == before
    pd.testing.assert_frame_equal(load_membership(first), result[['instance_id', 'specimen_id', 'source_dataset', 'label', 'split']])
    second = tmp_path / 'second'
    catalog.sample(frac=1, random_state=3).to_csv(paths[0], index=False)
    mapping.sample(frac=1, random_state=7).to_csv(paths[1], index=False)
    write_specimen_splits(*paths, second, candidates=32)
    for name in ['train.csv', 'validation.csv', 'test.csv', 'membership.csv', 'summary.csv', 'specimens.csv']:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    with pytest.raises(FileExistsError):
        write_specimen_splits(*paths, first, candidates=32)
    (first / 'membership.csv').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        load_membership(first)


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'null_group'])
def test_invalid_mapping_rejected(problem):
    catalog, mapping = synthetic()
    if problem == 'missing':
        mapping = mapping.iloc[:-1]
    elif problem == 'duplicate':
        mapping.loc[1, 'instance_id'] = mapping.loc[0, 'instance_id']
    else:
        mapping.loc[0, 'specimen_id'] = None
    with pytest.raises(ValueError):
        eligible_catalog(catalog, mapping)


def test_reference_exclusion_and_group_overlap():
    catalog, mapping = synthetic()
    reference = pd.DataFrame([['messreihe_1_x-28_y102', 3, 'messreihe_1', -28, 102]], columns=catalog.columns)
    reference_mapping = pd.DataFrame([['messreihe_1_x-28_y102', 'messreihe_1_pk_15']], columns=mapping.columns)
    eligible = eligible_catalog(pd.concat([catalog, reference], ignore_index=True),
                                pd.concat([mapping, reference_mapping], ignore_index=True))
    assert set(eligible.instance_id) == set(catalog.instance_id)
    assignments, _ = choose_groups(eligible, candidates=16)
    eligible['split'] = eligible.specimen_id.map(assignments)
    first = eligible.specimen_id.iloc[0]
    row = eligible.index[eligible.specimen_id.eq(first)][0]
    eligible.loc[row, 'split'] = 'test' if eligible.loc[row, 'split'] != 'test' else 'train'
    with pytest.raises(ValueError, match='overlap'):
        validate_membership(eligible)
