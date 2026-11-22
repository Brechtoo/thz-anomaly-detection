import numpy as np
import pytest

from src.features.thz_physical import PHYSICAL_COLUMNS, _haar_energy, extract_physical


def _trace(dt=0.05, echoes=()):
    x = np.linspace(960.0, 1160.0, round(200 / dt) + 1)

    def bipolar(center, width):
        t = (x - center) / width
        pulse = t * np.exp(-0.5 * t**2)
        pulse[np.abs(t) > 8] = 0
        return pulse

    y = 3.0 + bipolar(1000.0, 0.5)
    for delay, gain, width in echoes:
        y += gain * bipolar(1000 + delay, width)
    return x, y


def test_same_trace_echo_alignment_recovers_polarity_and_separation():
    x, y = _trace(echoes=[(20.0, -0.22, 0.5), (46.0, 0.16, 0.5)])
    result = extract_physical(x, y)
    assert tuple(result) == PHYSICAL_COLUMNS
    assert result["ncc__primary_signed"] == pytest.approx(-1)
    assert result["ncc__secondary_to_primary"] == pytest.approx(1)
    assert result["ncc__peak_separation_over_reference"] == pytest.approx(26 / 7.05)
    assert result["morph__echo_surface_fwhm_ratio"] == pytest.approx(1)
    assert result["morph__echo_surface_bipolar_span_ratio"] == pytest.approx(1)
    assert result["fft__log_echo_surface_amplitude_b1"] == pytest.approx(np.log(0.22), abs=0.05)
    assert np.isnan(result["ncc__residual_roughness"])


def test_shape_relative_features_are_invariant_to_uniform_gain_and_offset():
    x, y = _trace(echoes=[(20.0, -0.23, 0.7)])
    result = extract_physical(x, y)
    shifted = extract_physical(x, 3 * y + 42)
    np.testing.assert_allclose(list(result.values()), list(shifted.values()), rtol=1e-8, atol=1e-9)
    inverted = extract_physical(x, -3 * y + 42)
    assert inverted["ncc__primary_signed"] == pytest.approx(result["ncc__primary_signed"])
    assert inverted["morph__echo_minus_surface_polarity_balance"] == pytest.approx(
        -result["morph__echo_minus_surface_polarity_balance"],
    )
    assert result["morph__echo_surface_fwhm_ratio"] > 1.10
    assert result["ncc__residual_roughness"] > 0


def test_flat_post_keeps_columns_and_undefined_comparisons_as_nan():
    x, y = _trace()
    result = extract_physical(x, y)
    assert tuple(result) == PHYSICAL_COLUMNS
    assert result["ncc__match_count_per_reference_length"] == 0
    assert np.isnan(result["ncc__primary_signed"])
    assert np.isnan(result["morph__echo_surface_fwhm_ratio"])
    assert np.isnan(result["fft__log_echo_surface_amplitude_b1"])
    assert np.isnan(result["haar__log_post_surface_rms_l1"])


def test_sampling_interval_controls_template_spacing_and_bad_grids_are_rejected():
    coarse_x, coarse_y = _trace(dt=0.1, echoes=[(20.0, -0.22, 0.5), (46.0, 0.16, 0.5)])
    coarse = extract_physical(coarse_x, coarse_y)
    assert coarse["ncc__peak_separation_over_reference"] == pytest.approx(26 / 7.1)
    coarse_x[20] += 0.02
    with pytest.raises(ValueError, match="nonuniform_trace"):
        extract_physical(coarse_x, coarse_y)


def test_valid_haar_details_vanish_on_constant_and_localize_a_step():
    assert np.count_nonzero(_haar_energy(np.full(80, 0.175), 8)) == 0
    step = np.r_[np.zeros(40), np.ones(40)]
    for width in (1, 2, 4, 8, 16):
        power = _haar_energy(step, width)
        assert np.argmax(power) + width == 40
        assert power.max() == pytest.approx(width / 2)
