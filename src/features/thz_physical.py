"""Per-trace, early-anchor relative features for THz reflection measurements.

The recorded x axis defines all time and frequency scales. No cohort statistics,
material constants, calibrated physical units or external templates enter the fit.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks, hilbert, peak_widths

from src.features.experimental_anchors import adapted_start, early_index

BAND_EDGES = (0.125, 0.4, 0.8, 1.4, 2.4, 4.0)
LEVELS = 5
LEFT, RIGHT, POST_END = 2.5, 4.5, 80.0
MIN_NCC, MIN_PROMINENCE, MIN_SEPARATION = 0.45, 0.08, 2.0
MIN_RMS_FRACTION = 0.02
MIN_BAND_FRACTION = 1e-6
POWER_FLOOR_FRACTION = 1e-12

PHYSICAL_CONFIG = {
    "anchor": "experimental_anchors.early_index",
    "post_start": "experimental_anchors.adapted_start; strictly after reference window",
    "reference_window_x": [-LEFT, RIGHT],
    "post_end_x": POST_END,
    "fft_band_edges_per_x_unit": BAND_EDGES,
    "spectral_frame": "reference-length Hann frames, 50% overlap; near/far by post-span midpoint",
    "spectral_power_gate_fraction": MIN_BAND_FRACTION,
    "spectral_reference_power_floor_fraction": POWER_FLOOR_FRACTION,
    "haar_detail_levels": LEVELS,
    "haar_boundary": "valid undecimated filters, no padding",
    "template": "DC-centered surface window from the same measurement",
    "ncc_abs_threshold": MIN_NCC,
    "ncc_prominence": MIN_PROMINENCE,
    "ncc_peak_separation_x": MIN_SEPARATION,
    "ncc_minimum_window_rms_over_reference": MIN_RMS_FRACTION,
    "units": "recorded x units and reciprocal x units; no assumed ps or Hz",
}

FFT_COLUMNS = (
    *(f"fft__log_echo_surface_amplitude_b{i}" for i in range(1, len(BAND_EDGES))),
    *(f"fft__log_far_near_amplitude_b{i}" for i in range(1, len(BAND_EDGES))),
    "fft__echo_surface_band_distribution_distance",
    "fft__far_near_band_distribution_distance",
    "fft__high_band_temporal_shift",
    "fft__high_low_band_temporal_overlap",
)
HAAR_COLUMNS = (
    *(f"haar__log_post_surface_rms_l{i}" for i in range(1, LEVELS + 1)),
    "haar__post_minus_surface_scale_entropy",
    "haar__post_fine_coarse_envelope_overlap",
    "haar__post_fine_coarse_temporal_shift",
    "haar__post_fine_coarse_width_ratio",
    "haar__post_multiscale_hot_fraction",
)
MORPH_COLUMNS = (
    "morph__post_envelope_crest",
    "morph__echo_surface_fwhm_ratio",
    "morph__echo_minus_surface_half_height_asymmetry",
    "morph__echo_surface_bipolar_span_ratio",
    "morph__echo_minus_surface_polarity_balance",
    "morph__echo_surface_curvature_ratio",
)
NCC_COLUMNS = (
    "ncc__match_count_per_reference_length",
    "ncc__primary_signed",
    "ncc__half_prominence_width_over_reference",
    "ncc__secondary_to_primary",
    "ncc__peak_separation_over_reference",
    "ncc__residual_energy_edge_fraction",
    "ncc__residual_roughness",
)
PHYSICAL_COLUMNS = FFT_COLUMNS + HAAR_COLUMNS + MORPH_COLUMNS + NCC_COLUMNS


def _distribution(weights):
    total = weights.sum()
    return weights / total if total > 0 else np.full_like(weights, np.nan)


def _overlap(one, other):
    """Distribution overlap in [0, 1], undefined for zero total energy."""
    return float(np.minimum(_distribution(one), _distribution(other)).sum())


def _log_amplitude_ratio(num, den, min_num=0, min_den=0):
    if num <= min_num or den <= min_den:
        return np.nan
    return float(0.5 * (np.log(num) - np.log(den)))


def _spectrum(reference, post, dt):
    """Band-limited ratios and time-frequency distributions on matched Hann windows."""
    length = len(reference)
    params = np.fft.rfftfreq(length, d=dt)
    lower, upper = np.asarray(BAND_EDGES[:-1]), np.asarray(BAND_EDGES[1:])
    bands = (params[:, None] >= lower) & (params[:, None] < upper)
    bands[:, upper > 0.5 / dt] = False  # never report a partially observable band
    active = bands.any(axis=0)
    stride = max(length // 2, 1)
    frames = sliding_window_view(post, length)[::stride]
    windows = np.vstack([reference, frames])
    windows = (windows - windows.mean(axis=1, keepdims=True)) * np.hanning(length)
    power = np.abs(np.fft.rfft(windows, axis=1)) ** 2
    band_power = power @ bands
    ref, echo = band_power[0], band_power[1:]
    power_floor = POWER_FLOOR_FRACTION * ref.sum()
    strongest = echo[int(np.argmax(echo.sum(axis=1)))]
    midpoint = (len(post) - 1) / 2
    centers = np.arange(len(frames)) * stride + (length - 1) / 2
    time_fraction = centers / (len(post) - 1)
    near_mask = centers < midpoint
    near = echo[near_mask].mean(axis=0) if near_mask.any() else np.zeros_like(ref)
    far = echo[~near_mask].mean(axis=0) if (~near_mask).any() else np.zeros_like(ref)
    groups = [(strongest, ref), (far, near)]
    values = [
        _log_amplitude_ratio(
            a, b, max(power_floor, MIN_BAND_FRACTION * num.sum()),
            max(power_floor, MIN_BAND_FRACTION * den.sum()),
        )
        if covered else np.nan
        for num, den in groups
        for a, b, covered in zip(num, den, active, strict=True)
    ]
    values.extend(1 - _overlap(a, b) if min(a.sum(), b.sum()) > power_floor else np.nan for a, b in groups)
    high = echo[:, 3:].sum(axis=1)
    low = echo[:, :2].sum(axis=1)
    if min(high.sum(), low.sum()) > max(power_floor, MIN_BAND_FRACTION * echo.sum()):
        values.extend([np.dot(time_fraction, _distribution(high) - _distribution(low)), _overlap(high, low)])
    else:
        values.extend([np.nan, np.nan])
    return values


def _haar_energy(values, half_width):
    """Squared stationary Haar detail at each valid center, at native sample spacing."""
    sums = np.r_[0.0, np.cumsum(values - values[0])]
    left = sums[half_width:-half_width] - sums[:-2 * half_width]
    right = sums[2 * half_width:] - sums[half_width:-half_width]
    return (right - left) ** 2 / (2 * half_width)


def _haar_details(reference, post):
    """Relative scale energies and within-trace detail localization."""
    widths = 2 ** np.arange(LEVELS)
    reference_power = np.array([_haar_energy(reference, w).mean() for w in widths])
    details = [_haar_energy(post, w) for w in widths]
    post_power = np.array([detail.mean() for detail in details])
    values = [_log_amplitude_ratio(p, r) for p, r in zip(post_power, reference_power, strict=True)]

    def entropy(power):
        prob = _distribution(power)
        if not np.isfinite(prob).all():
            return np.nan
        nonzero = prob > 0
        return float(-np.sum(prob[nonzero] * np.log(prob[nonzero])) / np.log(LEVELS))

    values.append(entropy(post_power) - entropy(reference_power))
    # Compare at common valid centers, without padding either edge.
    common = [detail[widths[-1] - w:len(detail) - (widths[-1] - w)]
              for detail, w in zip(details, widths, strict=True)]
    fine, coarse = np.mean(common[:2], axis=0), np.mean(common[-2:], axis=0)
    time = (np.arange(len(fine)) + widths[-1] - 0.5) / (len(post) - 1)
    p_fine, p_coarse = _distribution(fine), _distribution(coarse)
    mean_fine, mean_coarse = np.dot(time, p_fine), np.dot(time, p_coarse)
    width_fine = np.sqrt(np.dot((time - mean_fine) ** 2, p_fine))
    width_coarse = np.sqrt(np.dot((time - mean_coarse) ** 2, p_coarse))
    concentrated = np.array([d > np.quantile(d, 0.90) + 1e-10 * d.max() for d in common])
    values.extend([
        _overlap(fine, coarse),
        mean_fine - mean_coarse,
        width_fine / width_coarse if width_coarse > 1e-8 else np.nan,
        np.mean(concentrated.sum(axis=0) >= 3),
    ])
    return values


def _pulse_shape(pulse):
    """Half-height Hilbert-envelope width/asymmetry and bipolar waveform summaries."""
    pulse = pulse - pulse.mean()
    envelope = np.abs(hilbert(pulse * np.hanning(len(pulse))))
    i = int(np.argmax(envelope))
    half_height = envelope[i] / 2
    left = np.flatnonzero(envelope[:i] < half_height)
    right = np.flatnonzero(envelope[i + 1:] < half_height)
    width, asymmetry = np.nan, np.nan
    if len(left) and len(right):
        lo, hi = left[-1], i + 1 + right[0]
        rise = i - lo - (half_height - envelope[lo]) / (envelope[lo + 1] - envelope[lo])
        fall = hi - i - (half_height - envelope[hi]) / (envelope[hi - 1] - envelope[hi])
        width = rise + fall
        asymmetry = (fall - rise) / width
    hi, lo = int(np.argmax(pulse)), int(np.argmin(pulse))
    balance = (pulse[hi] + pulse[lo]) / (pulse[hi] - pulse[lo]) if pulse[hi] > pulse[lo] else np.nan
    energy = np.dot(pulse, pulse)
    curvature = np.dot(np.diff(pulse, n=2), np.diff(pulse, n=2)) / energy if energy > 0 else np.nan
    return width, asymmetry, abs(hi - lo), balance, curvature


def _correlation(reference, post, dt):
    """Find plausible shifted copies of the surface pulse and measure residual shape."""
    length = len(reference)
    template = reference - reference.mean()
    template_norm = np.linalg.norm(template)
    norm_gate = MIN_RMS_FRACTION * template_norm
    post = post - np.median(post)
    cumulative = np.r_[0.0, np.cumsum(post)]
    squared = np.r_[0.0, np.cumsum(post**2)]
    sums = cumulative[length:] - cumulative[:-length]
    norms = np.sqrt(np.maximum(0, squared[length:] - squared[:-length] - sums**2 / length))
    ncc = np.zeros_like(norms)
    np.divide(np.correlate(post, template, mode="valid"), template_norm * norms,
              out=ncc, where=(norms > norm_gate) & (template_norm > 0))
    np.clip(ncc, -1, 1, out=ncc)
    magnitude = abs(ncc)
    peaks, _ = find_peaks(
        magnitude, height=MIN_NCC, prominence=MIN_PROMINENCE,
        distance=max(1, round(MIN_SEPARATION / dt)),
    )
    values = [len(peaks) * length / len(post), *([np.nan] * (len(NCC_COLUMNS) - 1))]
    if not len(peaks):
        return values, None
    ordered = peaks[np.argsort(-magnitude[peaks], kind="stable")]
    best = ordered[0]
    values[1] = ncc[best]
    values[2] = peak_widths(magnitude, [best], rel_height=0.5)[0][0] / length
    if len(ordered) > 1:
        second = ordered[1]
        values[3] = magnitude[second] / magnitude[best]
        values[4] = abs(second - best) / length
    echo = post[best:best + length]
    centered = echo - echo.mean()
    residual = centered - template * (np.dot(centered, template) / template_norm**2)
    residual_energy = np.dot(residual, residual)
    if residual_energy > 1e-12 * np.dot(centered, centered):
        radius = length // 4
        values[5] = (np.dot(residual[:radius], residual[:radius])
                     + np.dot(residual[-radius:], residual[-radius:])) / residual_energy
        values[6] = np.dot(np.diff(residual), np.diff(residual)) / residual_energy
    return values, echo * np.sign(ncc[best])


def extract_physical(x, y):
    """Return a fixed-schema feature mapping for one uniformly sampled trace."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 1 or y.shape != x.shape or len(x) < 40
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("invalid_trace")
    dt = (x[-1] - x[0]) / (len(x) - 1)
    if dt <= 0 or not np.allclose(np.diff(x), dt, atol=dt * 1e-5, rtol=0):
        raise ValueError("nonuniform_trace")
    anchor = early_index(x, y)
    left, right = round(LEFT / dt), round(RIGHT / dt)
    ref_start, ref_end = anchor - left, anchor + right + 1
    post_start = max(ref_end, anchor + int(np.ceil(adapted_start(x, y, anchor) / dt)))
    post_end = min(len(x), anchor + int(np.floor(POST_END / dt)) + 1)
    if ref_start < 0 or ref_end > len(y) or left + right + 1 < 2 ** LEVELS + 1:
        raise ValueError("incomplete_surface_window")
    if post_end - post_start < 2 * (left + right + 1):
        raise ValueError("incomplete_post_window")
    if post_end < anchor + int(np.floor(POST_END / dt)) + 1:
        raise ValueError("post_window_truncated")
    # Stable signal-wise offset and scale; all outputs compare like quantities.
    y = y - np.median(y)
    amplitude = np.max(np.abs(y[ref_start:ref_end]))
    if amplitude <= 0:
        raise ValueError("flat_surface_window")
    reference = y[ref_start:ref_end] / amplitude
    post = y[post_start:post_end] / amplitude
    spectrum = _spectrum(reference, post, dt)
    multiscale = _haar_details(reference, post)
    ncc, echo = _correlation(reference, post, dt)
    envelope = np.abs(hilbert(post - post.mean()))
    envelope_rms = np.sqrt(np.mean(envelope**2))
    morphology = [envelope.max() / envelope_rms if envelope_rms > 0 else np.nan, *([np.nan] * 5)]
    if echo is not None:
        ref_width, ref_asym, ref_span, ref_balance, ref_curvature = _pulse_shape(reference)
        echo_width, echo_asym, echo_span, echo_balance, echo_curvature = _pulse_shape(echo)
        morphology[1:] = [
            echo_width / ref_width if ref_width > 0 else np.nan,
            echo_asym - ref_asym,
            echo_span / ref_span if ref_span > 0 else np.nan,
            echo_balance - ref_balance,
            echo_curvature / ref_curvature if ref_curvature > 1e-12 else np.nan,
        ]
    return {name: float(value) if np.isfinite(value) else np.nan
            for name, value in zip(PHYSICAL_COLUMNS, spectrum + multiscale + morphology + ncc, strict=True)}
