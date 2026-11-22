import numpy as np
import pytest

from src.features.thz_echo_candidates import (
    candidates,
    correlation_curve,
    nonoverlapping_second,
)
from src.features.thz_physical_v2 import candidate_features


def test_v2_keeps_the_highest_ranked_disjoint_candidate():
    assert nonoverlapping_second(np.array([100, 40, 180, 241, 350]), 141) == 241
    assert nonoverlapping_second(np.array([100, 40, 180]), 141) is None
    assert nonoverlapping_second(np.array([], dtype=int), 141) is None


def test_one_oscillatory_pulse_can_produce_multiple_v1_peaks():
    dx = .05
    x = np.arange(-2.5, 4.5 + dx / 2, dx)
    ref = np.cos(2 * np.pi * .45 * x) * np.exp(-.5 * (x / 1.3) ** 2)
    signal = np.zeros(1200)
    signal[400:400+len(ref)] = .15 * ref
    score, rms = correlation_curve(ref, signal)
    peaks = candidates(score, rms, dx)
    assert len(peaks) > 1
    assert peaks[0] == 400
    assert nonoverlapping_second(peaks, len(ref)) is None


def test_separated_returns_are_retained_and_empty_post_is_undefined():
    x = np.linspace(960, 1160, 4001)

    def pulse(center):
        t = (x - center) / .5
        return t * np.exp(-.5 * t**2)

    surface = 4 + pulse(1000)
    empty, density, axis = candidate_features(x, surface)
    assert density == empty[3] == 0
    assert np.isnan(empty[:3]).all()
    assert axis == (4001, .05)

    output, density, _ = candidate_features(x, surface - .2 * pulse(1020) + .15 * pulse(1046))
    assert density > 0
    assert output[0] == pytest.approx(1)
    assert output[1] == pytest.approx(26 / 7.05)
    assert output[2] == pytest.approx(1)
    assert 0 < output[3] < 1
