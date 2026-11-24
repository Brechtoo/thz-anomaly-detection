import numpy as np
import pandas as pd
import pytest

from src.evaluation.spatial_postprocessing import (
    NeighborSummary,
    SpatialCandidate,
    apply_strategy,
    build_neighborhoods,
    candidate_grid,
    summarize_neighbors,
)


def test_radius_respects_physical_gaps_and_surfaces():
    points = pd.DataFrame({
        'specimen_id': ['a', 'b', 'a', 'a', 'a', 'a', 'a'],
        'source_dataset': ['one', 'one', 'one', 'one', 'two', 'one', 'one'],
        'x': [0, 0, 1, 2, 0, 10, 11],
        'y': [0, 0, 0, 0, 0, 0, 0],
    }, index=[10, 11, 13, 17, 19, 21, 23])

    small, large = build_neighborhoods(points, [1.5, 2.5]).values()
    np.testing.assert_array_equal(small.counts, [1, 0, 2, 1, 0, 1, 1])
    np.testing.assert_array_equal(large.counts, [2, 0, 2, 2, 0, 1, 1])
    assert large.distance[0, 3] == 2
    assert large.adjacency[0, 1] == large.adjacency[0, 4] == large.adjacency[3, 5] == 0
    assert (large.adjacency != large.adjacency.T).nnz == 0
    assert large.adjacency.diagonal().sum() == 0


def test_gaussian_weights_actual_distance_and_normalizes_at_edges():
    points = pd.DataFrame({'specimen_id': ['a'] * 3, 'source_dataset': ['one'] * 3,
                           'x': [0., 1., 2.5], 'y': [0., 0., 0.]})
    graph = build_neighborhoods(points, [2.5])[2.5]
    probabilities = np.eye(4)[[1, 3, 2]]
    uniform = summarize_neighbors(graph, probabilities, 'uniform')
    gaussian = summarize_neighbors(graph, probabilities, 'gaussian')

    np.testing.assert_allclose(uniform.mean[1], [0., .5, .5, 0.])
    assert gaussian.mean[1, 1] > gaussian.mean[1, 2]
    np.testing.assert_allclose(gaussian.mean.sum(axis=1), 1.)
    np.testing.assert_allclose(gaussian.votes, gaussian.mean)


def test_geometry_rejects_reference_regions_and_duplicate_coordinates():
    points = pd.DataFrame({'specimen_id': ['a', 'a'], 'source_dataset': ['messreihe_1'] * 2,
                           'x': [-20, -20], 'y': [105, 106]})
    with pytest.raises(ValueError, match='reference measurements'):
        build_neighborhoods(points, [1.5])

    points['y'] = [20, 20]
    with pytest.raises(ValueError, match='Duplicate X/Y'):
        build_neighborhoods(points, [1.5])


@pytest.mark.parametrize('strategy,strength,threshold', [
    ('smooth', .50, .02),
    ('rescue', .12, .5),
    ('consensus', .25, .55),
])
def test_spatial_rules_preserve_isolated_and_strong_evidence(strategy, strength, threshold):
    probabilities = np.array([
        [.04, .71, .10, .15],
        [.02, .10, .60, .28],
        [.02, .39, .18, .41],
        [.02, .39, .18, .41],
        [.02, .35, .24, .39],
    ])
    baseline = np.array([1, 2, 3, 3, 1])
    surroundings = np.array([
        [.00, .00, .00, 1.],
        [1., .00, .00, .00],
        [.00, .90, .05, .05],
        [.00, .90, .05, .05],
        [.00, .00, .00, 1.],
    ])
    summary = NeighborSummary(surroundings, np.eye(4)[surroundings.argmax(axis=1)], np.array([3, 3, 2, 1, 4]))
    candidate = SpatialCandidate(strategy, 1.5, 'uniform', strength, threshold)
    np.testing.assert_array_equal(apply_strategy(baseline, probabilities, summary, candidate), [1, 2, 1, 3, 1])
    np.testing.assert_array_equal(baseline, [1, 2, 3, 3, 1])


def test_pointwise_control_needs_no_geometry_and_preserves_other_classes():
    probabilities = np.array([[.2, .1, .1, .6], [.1, .1, .1, .7], [.1, .1, .1, .7], [.4, .1, .1, .4]])
    candidate = SpatialCandidate('pointwise', None, None, 0., .65)
    np.testing.assert_array_equal(apply_strategy([3, 3, 1, 0], probabilities, None, candidate), [2, 3, 1, 0])
    assert len(set(candidate_grid())) == 79
