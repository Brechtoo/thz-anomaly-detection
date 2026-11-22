"""Window and candidate diagnostics shared by the V1 audit and V2 features."""
import numpy as np
from scipy.signal import find_peaks

from src.features import thz_physical as v1
from src.features.experimental_anchors import adapted_start, early_index
from src.features.extract_specimen_features import uniform_trace


def correlation_curve(reference, post):
    """Recompute V1's local Pearson correlation and relative window RMS before gating."""
    template = reference - reference.mean()
    norm = np.linalg.norm(template)
    post = post - np.median(post)
    length = len(template)
    cumulative = np.r_[0.0, np.cumsum(post)]
    squared = np.r_[0.0, np.cumsum(post**2)]
    sums = cumulative[length:] - cumulative[:-length]
    energy = squared[length:] - squared[:-length] - sums**2 / length
    window_norm = np.sqrt(np.maximum(0, energy))
    ratio = window_norm / norm
    scores = np.zeros_like(ratio)
    np.divide(np.correlate(post, template, mode='valid'), norm * window_norm,
              out=scores, where=window_norm > 0)
    return np.clip(scores, -1, 1), ratio


def candidates(scores, rms, dx, *, height=v1.MIN_NCC, prominence=v1.MIN_PROMINENCE,
               gate=v1.MIN_RMS_FRACTION):
    gated = np.where(rms > gate, scores, 0)
    peaks, _ = find_peaks(np.abs(gated), height=height, prominence=prominence,
                          distance=max(1, round(v1.MIN_SEPARATION / dx)))
    return peaks[np.argsort(-np.abs(gated[peaks]), kind='stable')]


def nonoverlapping_second(peaks, template_samples):
    """Second V1 candidate with a full window disjoint from the top candidate."""
    if len(peaks):
        alternatives = peaks[1:][np.abs(peaks[1:] - peaks[0]) >= template_samples]
        if len(alternatives):
            return int(alternatives[0])
    return None


def trace_windows(x, y):
    x, y, dx, _ = uniform_trace(x, y)
    anchor = early_index(x, y)
    left, right = round(v1.LEFT / dx), round(v1.RIGHT / dx)
    lo, hi = anchor - left, anchor + right + 1
    start = max(hi, anchor + int(np.ceil(adapted_start(x, y, anchor) / dx)))
    end = min(len(y), anchor + int(np.floor(v1.POST_END / dx)) + 1)
    y = y - np.median(y)
    amp = np.max(np.abs(y[lo:hi]))
    return x - x[anchor], y / amp, dx, lo, hi, start, end
