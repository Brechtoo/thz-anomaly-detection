"""Time-, distribution-, and frequency-domain TSFEL features on a full trace."""
from copy import deepcopy

import numpy as np

DOMAINS = ('statistical', 'temporal', 'spectral')
OMIT = {
    'statistical': {
        'Absolute energy', 'Average power', 'ECDF', 'ECDF Percentile Count', 'Kurtosis',
        'Max', 'Mean', 'Median', 'Min', 'Peak to peak distance', 'Root mean square',
        'Skewness', 'Standard deviation', 'Variance',
    },
    'temporal': {
        'Mean absolute diff', 'Mean diff', 'Neighbourhood peaks', 'Positive turning points',
        'Slope', 'Sum absolute diff',
    },
    'spectral': {
        'Human range energy', 'MFCC', 'LPCC', 'Fundamental frequency', 'Maximum frequency',
        'Wavelet variance', 'Wavelet standard deviation', 'Wavelet absolute mean',
    },
}


class TsfelExtractor:
    """Use TSFEL's individual calculators so undefined components stay missing."""

    def __init__(self):
        import tsfel

        self.config = deepcopy(tsfel.get_features_by_domain(list(DOMAINS)))
        self.omitted = {}
        for domain, features in self.config.items():
            for name, settings in features.items():
                if name in OMIT[domain] or settings['use'] != 'yes':
                    settings['use'] = 'no'
                    self.omitted[name] = domain

        # A non-degenerate synthetic signal fixes the schema before reading any cohort.
        t = np.arange(4001, dtype=float) / 20
        probe = np.sin(2 * np.pi * 0.7 * t) * np.exp(-((t - 35) / 3) ** 2)
        probe += 0.01 * np.sin(2 * np.pi * 2.1 * t)
        self.calculators = []
        for domain in DOMAINS:
            for name, settings in self.config[domain].items():
                if settings['use'] != 'yes':
                    continue
                calculator = getattr(tsfel, settings['function'].removeprefix('tsfel.'))
                params = settings['parameters'] or {}
                values = self._call(calculator, probe, 20.0, params)
                slug = name.lower().replace(' ', '_').replace('-', '_')
                names = (f'{domain}__{slug}',) if len(values) == 1 else tuple(
                    f'{domain}__{slug}__{i:02d}' for i in range(len(values))
                )
                self.calculators.append((calculator, params, names))
        self.columns = tuple(name for _, _, names in self.calculators for name in names)

    @staticmethod
    def _call(calculator, y, fs, params):
        params = dict(params)
        if 'fs' in params:
            params['fs'] = fs
        value = calculator(y, **params)
        # Frequency-bin vectors are named by index. Physical coordinates are
        # determined by per-trace sampling metadata, not the library's Hz labels.
        if isinstance(value, dict):
            value = value['values']
        return np.atleast_1d(np.asarray(value, dtype=float))

    def extract(self, y, fs):
        y = y - np.median(y)
        if np.ptp(y) == 0:
            raise ValueError('flat_trace')
        result, errors = {}, []
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            for calculator, params, names in self.calculators:
                try:
                    values = self._call(calculator, y, fs, params)
                    if values.shape != (len(names),):
                        raise ValueError('changed_output_shape')
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                    values = np.full(len(names), np.nan)
                    errors.append(f'{calculator.__name__}: {type(exc).__name__}')
                else:
                    if not np.isfinite(values).all():
                        errors.append(f'{calculator.__name__}: nonfinite')
                result.update(zip(names, values, strict=True))
        return result, errors
