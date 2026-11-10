import numpy as np
import pytest
from tsfresh.feature_extraction import feature_calculators as fc

from src.analysis.feature_overlap import IDENTICAL_FORMULAS
from src.features.extract_baseline_features import extract_baseline_features


@pytest.mark.parametrize('seed', [0, 17, 42])
def test_duplicate_formulas_on_synthetic_signals(tmp_path, seed):
    y = np.random.default_rng(seed).normal(size=257)
    path = tmp_path / 'signal.txt'
    np.savetxt(path, np.column_stack([np.arange(len(y)), y]))
    baseline = extract_baseline_features(path)
    assert len(IDENTICAL_FORMULAS) == 6
    for key, tsname in IDENTICAL_FORMULAS.items():
        actual = getattr(fc, tsname.removeprefix('y__'))(y)
        assert baseline[key] == pytest.approx(actual, rel=1e-12, abs=1e-12)
