"""Spatial decisions from four-class probabilities on one measured surface at a time."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from src.data.reference_measurements import require_no_references
from src.evaluation.svm_nested_selection import apply_policy

RADIUS_FACTORS = (1.5, 2.5, 4.)
WEIGHTINGS = ('uniform', 'gaussian')


@dataclass(frozen=True)
class Neighborhood:
    adjacency: csr_matrix
    distance: csr_matrix
    counts: np.ndarray
    spacing: np.ndarray
    radius_factor: float


@dataclass(frozen=True)
class NeighborSummary:
    mean: np.ndarray
    votes: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class SpatialCandidate:
    strategy: str
    radius_factor: float | None
    weighting: str | None
    strength: float
    threshold: float
    min_neighbors: int = 2
    preserve_probability: float = .65


def candidate_grid():
    """Fix the search space before inspecting outer-fold predictions."""
    candidates = []
    settings = {
        'smooth': ((.25, .02), (.25, .08), (.5, .02), (.5, .08)),
        'rescue': ((.05, .35), (.05, .5), (.12, .35), (.12, .5)),
        'consensus': ((.1, .55), (.1, .75), (.25, .55), (.25, .75)),
    }
    for strategy, rules in settings.items():
        candidates.extend(
            SpatialCandidate(strategy, radius, weighting, strength, threshold)
            for radius, weighting, (strength, threshold) in product(RADIUS_FACTORS, WEIGHTINGS, rules)
        )
    candidates.extend(
        SpatialCandidate('pointwise', None, None, 0., threshold)
        for threshold in (.40, .50, .60, .70, .80, .90, .95)
    )
    return tuple(candidates)


def build_neighborhoods(points, radius_factors):
    """Build symmetric radius graphs in row order using median nearest-point spacing per surface."""
    groups = ['specimen_id', 'source_dataset']
    require_no_references(points[['source_dataset', 'x', 'y']])
    if points[groups].isna().any().any():
        raise ValueError('Missing specimen or source information')
    coordinates = points[['x', 'y']].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError('Spatial processing requires finite X/Y coordinates')
    factors = sorted(set(radius_factors))
    if any(not np.isfinite(factor) or factor <= 0 for factor in factors):
        raise ValueError('Radius factors must be positive and finite')
    edges = {factor: [] for factor in factors}
    spacing = np.full(len(points), np.nan)
    for indices in points.groupby(groups, sort=False).indices.values():
        if len(indices) < 2:
            continue
        xy = coordinates[indices]
        if len(np.unique(xy, axis=0)) != len(xy):
            raise ValueError('Duplicate X/Y positions within a specimen and source')
        tree = cKDTree(xy)
        typical_spacing = float(np.median(tree.query(xy, k=2)[0][:, 1]))
        spacing[indices] = typical_spacing
        if not factors:
            continue
        # Use measured distances. A gap leaves fewer neighbors rather than expanding the radius.
        pairs = tree.query_pairs(factors[-1] * typical_spacing, output_type='ndarray')
        distances = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
        for factor in factors:
            inside = distances <= factor * typical_spacing
            edges[factor].append((indices[pairs[inside, 0]], indices[pairs[inside, 1]], distances[inside]))

    result = {}
    size = (len(points), len(points))
    for factor, chunks in edges.items():
        left, right, distance = (np.concatenate(column) for column in zip(*chunks)) if chunks else ([], [], [])
        row = np.concatenate((left, right)).astype(np.intp)
        col = np.concatenate((right, left)).astype(np.intp)
        distance = csr_matrix((np.tile(distance, 2), (row, col)), shape=size)
        adjacency = distance.copy()
        adjacency.data.fill(1.)
        result[factor] = Neighborhood(adjacency, distance, np.diff(adjacency.indptr), spacing, factor)
    return result


def summarize_neighbors(neighborhood, probabilities, weighting):
    """Compute probability means and probability-argmax votes using only nearby predictions."""
    if weighting == 'uniform':
        weights = neighborhood.adjacency
    elif weighting == 'gaussian':
        weights = neighborhood.distance.copy()
        rows = np.repeat(np.arange(len(neighborhood.counts)), neighborhood.counts)
        sigma = .5 * neighborhood.radius_factor * neighborhood.spacing[rows]
        weights.data = np.exp(-.5 * (weights.data / sigma) ** 2)
    else:
        raise ValueError(f'Unknown spatial weighting {weighting!r}')
    support = np.asarray(weights.sum(axis=1)).ravel()
    denominator = np.where(support > 0, support, 1.)[:, None]
    hard = np.eye(4)[probabilities.argmax(axis=1)]
    return NeighborSummary(weights @ probabilities / denominator,
                           weights @ hard / denominator, neighborhood.counts)


def apply_strategy(baseline, probabilities, summary, candidate):
    """Apply a fixed rule. Labels are needed only to evaluate and select candidates later."""
    baseline = np.asarray(baseline, dtype=int)
    if candidate.strategy == 'pointwise':
        policy = {'gate': 'predict', 'threshold': candidate.threshold, 'fallback': 'class2'}
        return apply_policy(baseline, probabilities, policy)

    row = np.arange(len(baseline))
    own_abnormal = probabilities[:, 1:3].sum(axis=1)
    if candidate.strategy == 'smooth':
        pooled = (1 - candidate.strength) * probabilities + candidate.strength * summary.mean
        proposed = pooled.argmax(axis=1)
        update = ((pooled[row, proposed] - pooled[row, baseline] >= candidate.threshold)
                  & (summary.mean[row, proposed] > probabilities[row, proposed]))
    elif candidate.strategy == 'rescue':
        nearby_abnormal = summary.mean[:, 1:3].sum(axis=1)
        # Require both own evidence and a corroborating neighborhood. Use both to choose 1 or 2.
        proposed = 1 + (probabilities[:, 1:3] + summary.mean[:, 1:3]).argmax(axis=1)
        update = ((baseline == 3) & (own_abnormal >= .10) & (nearby_abnormal >= candidate.threshold)
                  & (nearby_abnormal - own_abnormal >= candidate.strength))
    elif candidate.strategy == 'consensus':
        proposed = summary.votes.argmax(axis=1)
        update = ((summary.votes[row, proposed] >= candidate.threshold)
                  & (probabilities[row, baseline] - probabilities[row, proposed] <= candidate.strength)
                  & (summary.mean[row, proposed] > probabilities[row, proposed]))
    else:
        raise ValueError(f'Unknown spatial strategy {candidate.strategy!r}')

    abnormal_baseline = np.isin(baseline, [1, 2])
    strong_evidence = own_abnormal >= candidate.preserve_probability
    protected = ((abnormal_baseline & strong_evidence)
                 | ((proposed == 3) & (abnormal_baseline | strong_evidence)))
    return np.where(update & (summary.counts >= candidate.min_neighbors) & ~protected, proposed, baseline)
