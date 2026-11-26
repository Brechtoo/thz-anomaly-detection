"""Signal context calculated from measured neighbors of the same specimen."""

import numpy as np
import pandas as pd

from src.evaluation.spatial_postprocessing import build_neighborhoods

RADII = (1.5, 4.0)
SIGNAL_CHANNELS = (
    'y_peak_to_peak',
    'first_third_std',
    'second_peak_amp',
    'second_peak_distance',
    'second_peak_snr',
    'second_peak_to_surface_ratio',
    'post_peak_std',
    'second_peak_surface_corr',
)
STATISTICS = ('mean', 'std', 'delta')
SPATIAL_KEYS = {'instance_id', 'source_dataset', 'specimen_id', 'x', 'y', 'z',
                'label', 'split', 'fold', 'cv_fold', 'file_path'}


class SpatialContext:
    """Summarize an acquired surface using fit-free, label-free neighbor graphs.

    Whole specimens can be transformed separately or together. All radius and
    weight calculations are internal to a specimen/source pair. Fold-specific
    missing-value imputation and scaling belong to the model pipeline.
    """

    def __init__(self, points, radii=RADII, weighting='gaussian'):
        if not points.index.is_unique:
            raise ValueError('Spatial locations require unique row indices')
        if weighting not in ('uniform', 'gaussian'):
            raise ValueError(f'Unknown spatial weighting: {weighting}')
        graphs = build_neighborhoods(points, radii)
        if not graphs:
            raise ValueError('At least one spatial radius is required')
        self.index = points.index.copy()
        self.weights = {}
        for radius, graph in graphs.items():
            if weighting == 'uniform':
                self.weights[radius] = graph.adjacency
                continue
            weights = graph.distance.copy()
            rows = np.repeat(np.arange(len(points)), graph.counts)
            # Use the same truncated Gaussian as the decision postprocessing.
            sigma = 0.5 * radius * graph.spacing[rows]
            weights.data = np.exp(-0.5 * (weights.data / sigma) ** 2)
            self.weights[radius] = weights

    def transform(self, features, radii=None, statistics=STATISTICS):
        """Return local mean, population SD and/or center-minus-mean for each channel."""
        if not features.index.equals(self.index):
            raise ValueError('Feature rows must match the spatial locations in index and order')
        if (features.empty or not features.columns.is_unique
                or SPATIAL_KEYS.intersection(features.columns)
                or not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes)):
            raise ValueError('Expected distinct numeric signal channels')
        radii = tuple(self.weights) if radii is None else tuple(radii)
        statistics = tuple(statistics)
        if (not radii or len(set(radii)) != len(radii) or not set(radii) <= self.weights.keys()
                or not statistics or len(set(statistics)) != len(statistics)
                or not set(statistics) <= set(STATISTICS)):
            raise ValueError('Expected distinct configured radii and spatial statistics')

        values = features.to_numpy(dtype=np.float64, na_value=np.nan)
        valid = np.isfinite(values)
        signals = np.where(valid, values, 0.0)
        output = {}
        for radius in radii:
            weights = self.weights[radius]
            support = weights @ valid
            means = np.divide(weights @ signals, support,
                              out=np.full(values.shape, np.nan), where=support > 0)
            summaries = {'mean': means, 'delta': np.where(valid, values, np.nan) - means}
            if 'std' in statistics:
                summaries['std'] = _standard_deviation(weights, signals, valid, support, means)
            for statistic in statistics:
                for column, measurements in zip(features.columns, summaries[statistic].T, strict=True):
                    output[f'spatial_r{radius:g}::{statistic}::{column}'] = measurements
        return pd.DataFrame(output, index=self.index)


def _standard_deviation(weights, signals, valid, support, means):
    """Use local residuals to avoid cancellation in nearly constant channels."""
    rows = np.repeat(np.arange(weights.shape[0]), np.diff(weights.indptr))
    columns = weights.indices
    result = np.full_like(means, np.nan)
    for i in range(signals.shape[1]):
        finite = valid[columns, i]
        neighbors = rows[finite]
        residual = signals[columns[finite], i] - means[neighbors, i]
        squared = np.bincount(neighbors, weights=weights.data[finite] * residual**2, minlength=len(means))
        enough = np.bincount(neighbors, minlength=len(means)) >= 2
        np.divide(squared, support[:, i], out=result[:, i], where=enough)
    return np.sqrt(result)
