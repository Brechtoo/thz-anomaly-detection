import numpy as np
import pandas as pd
import pytest

from src.features.spatial_context import SpatialContext


def points(x, *, body='body', source='source', index=None):
    return pd.DataFrame({'specimen_id': body, 'source_dataset': source, 'x': x, 'y': 0.}, index=index)


def test_center_and_neighbors_are_used_for_distinct_statistics():
    data = points([0., 1., 2.], index=['a', 'b', 'c'])
    signals = pd.DataFrame({'amplitude': [10., 40., 20.]}, index=data.index)
    result = SpatialContext(data, radii=(1.5,), weighting='uniform').transform(signals)

    assert result.index.equals(data.index)
    np.testing.assert_allclose(result.iloc[:, 0], [40., 15., 40.])
    np.testing.assert_allclose(result.iloc[:, 1], [np.nan, 5., np.nan])
    np.testing.assert_allclose(result.iloc[:, 2], [-30., 25., -20.])


def test_gaussian_and_actual_radius_with_irregular_spacing():
    data = points([0., 1., 2.4, 9.])
    signals = pd.DataFrame({'amplitude': [10., 42., 20., 999.]})
    result = SpatialContext(data, radii=(1.5, 4.)).transform(signals, statistics=('mean',))

    spacing = 1.2
    for radius in (1.5, 4.):
        sigma = radius * spacing / 2
        distance = np.array([1., 1.4])
        mean = np.average([10., 20.], weights=np.exp(-.5 * (distance / sigma)**2))
        assert result.loc[1, f'spatial_r{radius:g}::mean::amplitude'] == pytest.approx(mean)
    assert result.loc[0, 'spatial_r1.5::mean::amplitude'] == pytest.approx(42.)
    assert result.loc[0, 'spatial_r4::mean::amplitude'] < 42.
    assert result.iloc[-1].isna().all()


def test_transforming_unrelated_surfaces_cannot_change_a_body():
    a = points([1., 2., 3.], index=['a1', 'a2', 'a3'])
    b = points([1., 2., 3.], source='other', index=['b1', 'b2', 'b3'])
    c = points([1., 2., 3.], body='other', index=['c1', 'c2', 'c3'])
    combined = pd.concat([b, a, c])
    signals = pd.DataFrame({'signal': [100., 500., -200., 1., 5., 10., 50., 70., 99.]}, index=combined.index)

    alone = SpatialContext(a).transform(signals.loc[a.index])
    batch = SpatialContext(combined).transform(signals)
    pd.testing.assert_frame_equal(batch.loc[a.index], alone, check_exact=True)


def test_unknown_signal_values_do_not_become_neighbor_evidence():
    data = points([0., 1., 2.])
    signals = pd.DataFrame({'a': [3., np.nan, 7.], 'b': [np.inf, 2., np.nan]})
    result = SpatialContext(data, radii=(1.5,), weighting='uniform').transform(signals)

    assert result.loc[1, 'spatial_r1.5::mean::a'] == 5.
    assert result.loc[1, 'spatial_r1.5::std::a'] == 2.
    assert np.isnan(result.loc[1, 'spatial_r1.5::delta::a'])
    assert np.isnan(result.loc[1, 'spatial_r1.5::mean::b'])
    assert np.isnan(result.loc[0, 'spatial_r1.5::std::b'])


def test_alignment_and_metadata_validation():
    data = points([0., 1., 2.])
    context = SpatialContext(data)
    signals = pd.DataFrame({'a': [1., 2., 3.]})
    with pytest.raises(ValueError, match='index and order'):
        context.transform(signals.iloc[::-1])
    with pytest.raises(ValueError, match='numeric signal channels'):
        context.transform(signals.assign(label=0))
    with pytest.raises(ValueError, match='configured radii'):
        context.transform(signals, radii=(2.5,))
    with pytest.raises(ValueError, match='reference measurements'):
        SpatialContext(points([-20., -21., -22.], source='messreihe_1').assign(y=110.))
