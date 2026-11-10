import pandas as pd
import pytest

from src.data.dataset import EXCLUDE_COLUMNS, prepare_dataset
from src.data.specimen_mapping import identify_specimens, write_mapping


def synthetic():
    rows = []
    for source in ['a', 'b']:
        for offset in [0, 25]:
            for x in range(12):
                for y in range(12):
                    if (x, y) != (5, 5):  # internal measurement hole
                        rows.append([f'{source}_{x+offset}_{y}', source, x+offset, y])
        rows.append([f'{source}_outlier', source, 100, 100])
    return pd.DataFrame(rows, columns=['instance_id', 'source_dataset', 'x', 'y'])


def test_complete_unique_order_independent_and_holes():
    data = synthetic()
    before = data.copy(deep=True)
    mapping, summary = identify_specimens(data)
    shuffled, other = identify_specimens(data.sample(frac=1, random_state=42))
    pd.testing.assert_frame_equal(mapping, shuffled)
    pd.testing.assert_frame_equal(summary, other)
    pd.testing.assert_frame_equal(data, before)
    assert mapping.instance_id.is_unique
    assert set(mapping.instance_id) == set(data.instance_id)
    assert summary.groupby('source_dataset').size().tolist() == [2, 2]
    assert summary.n_points.tolist() == [143] * 4
    assert mapping.specimen_id.isna().sum() == 2


@pytest.mark.parametrize('kind', ['duplicate_id', 'duplicate_xy', 'nan', 'inf'])
def test_reject_invalid_metadata(kind):
    data = synthetic()
    if kind == 'duplicate_id':
        data.loc[1, 'instance_id'] = data.loc[0, 'instance_id']
    elif kind == 'duplicate_xy':
        data.loc[1, ['x', 'y']] = data.loc[0, ['x', 'y']]
    else:
        data['x'] = data.x.astype(float)
        data.loc[0, 'x'] = float(kind)
    with pytest.raises(ValueError):
        identify_specimens(data)


def test_small_region_requires_review():
    data = pd.DataFrame([[f'p_{x}_{y}', 'a', x, y]
                         for offset in [0, 25] for x in range(offset, offset + 20)
                         for y in range(20)], columns=['instance_id', 'source_dataset', 'x', 'y'])
    extra = pd.DataFrame([[f's_{x}_{y}', 'a', x, y] for x in range(50, 60) for y in range(10)],
                         columns=data.columns)
    # Increase primary regions so that 100-point region is below half median.
    large = pd.DataFrame([[f'l_{x}_{y}', 'a', x, y] for x in range(120, 140) for y in range(20)], columns=data.columns)
    mapping, _ = identify_specimens(pd.concat([data, extra, large]))
    assert set(mapping[mapping.instance_id.str.startswith('s_')].assignment_status) == {'review_small_region'}


def test_write_safe_and_metadata_not_feature(tmp_path):
    source = tmp_path / 'catalog.csv'
    synthetic().to_csv(source, index=False)
    before = source.read_bytes()
    output = tmp_path / 'metadata'
    mapping, _ = write_mapping(source, output)
    assert source.read_bytes() == before
    assert (output / 'a.png').exists()
    assert (output / 'manifest.json').exists()
    with pytest.raises(FileExistsError):
        write_mapping(source, output)
    data = pd.DataFrame({c: [0] for c in EXCLUDE_COLUMNS})
    data['feature'] = 1.0
    data['specimen_id'] = 'a_pk_01'
    X, _ = prepare_dataset(data)
    assert list(X) == ['feature']
