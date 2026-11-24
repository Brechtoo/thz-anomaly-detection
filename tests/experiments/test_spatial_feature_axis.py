from itertools import product

import numpy as np
import pandas as pd
import pytest

from src.experiments.spatial_feature_axis import RADIUS, CardinalContext
from src.features.spatial_context import SpatialContext


def test_cardinal_neighbors_drop_diagonals_preserving_gaussian_weights():
    points = pd.DataFrame(product(range(3), repeat=2), columns=['x', 'y'])
    points = points.assign(specimen_id='body', source_dataset='series')
    axial = CardinalContext(points)
    radial = SpatialContext(points, radii=(RADIUS,))
    weights = axial.weights[RADIUS]
    center = weights.getrow(4)
    np.testing.assert_array_equal(center.indices, [1, 3, 5, 7])
    np.testing.assert_array_equal(center.data, radial.weights[RADIUS][4, center.indices].toarray()[0])
    np.testing.assert_array_equal(weights.getrow(0).indices, [1, 3])


def test_cardinal_uses_nearest_in_each_direction_within_radius_and_group():
    points = pd.DataFrame({'x': [0., 1., 1.2, 2.2, 3.2, 4.2, 8.2, 12.2], 'y': 0,
                           'source_dataset': 'series', 'specimen_id': 'target'})
    other_source = points.iloc[:2].assign(source_dataset='other_series')
    other_specimen = points.iloc[:2].assign(specimen_id='other_body')
    locations = pd.concat([points, other_source, other_specimen], ignore_index=True)
    context = CardinalContext(locations)
    weights = context.weights[RADIUS]
    np.testing.assert_array_equal(weights.getrow(0).indices, [1])
    np.testing.assert_array_equal(weights.getrow(2).indices, [1, 3])
    assert weights.getrow(6).nnz == 0
    signal = pd.DataFrame({'s': np.arange(len(locations), dtype=float)})
    statistics = context.transform(signal, statistics=('delta', 'std'))
    assert statistics.loc[0, 'spatial_r1.5::delta::s'] == pytest.approx(-1.)
    assert np.isnan(statistics.loc[0, 'spatial_r1.5::std::s'])
    assert statistics.loc[6].isna().all()
