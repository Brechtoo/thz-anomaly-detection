import numpy as np
import pytest

from src.features.extract_specimen_features import uniform_trace
from src.features.specimen_tsfel import TsfelExtractor


def test_recorded_sampling_interval_and_small_jitter():
    x = 960 + 0.05 * np.arange(4001, dtype=float)
    y = 2 * x + 3
    assert uniform_trace(x, y)[2:] == (0.05, False)
    jittered = x.copy()
    jittered[1:-1:2] += 0.0001
    uniform, samples, dx, resampled = uniform_trace(jittered, 2 * jittered + 3)
    assert resampled and dx == 0.05
    np.testing.assert_allclose(uniform, x)
    np.testing.assert_allclose(samples, y)
    with pytest.raises(ValueError, match='gap'):
        uniform_trace(np.delete(x, 50), np.delete(y, 50))


def test_tsfel_frequencies_follow_recorded_axis_and_schema_is_stable():
    pytest.importorskip('tsfel')
    extractor = TsfelExtractor()
    t = np.arange(4001, dtype=float) / 20
    y = np.sin(2 * np.pi * 0.7 * t) * np.exp(-((t - 35) / 3) ** 2)
    features, errors = extractor.extract(y, 20.0)
    compressed, other_errors = extractor.extract(y, 40.0)
    assert not errors and not other_errors
    assert tuple(features) == tuple(compressed) == extractor.columns
    name = 'spectral__spectral_centroid'
    assert compressed[name] == pytest.approx(2 * features[name])
    assert set(name.split('__')[0] for name in features) == {'statistical', 'temporal', 'spectral'}
    assert {'ECDF', 'Neighbourhood peaks', 'Positive turning points', 'Maximum frequency'} <= set(extractor.omitted)
    assert np.isfinite(list(features.values())).all()
