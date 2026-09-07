import numpy as np

from scipy.signal import find_peaks

from src.features.loading import load_measurement


# ============================================================
# settings
# ============================================================

POST_PEAK_START = 3
NOISE_REGION_START = 70
END_OF_SIGNAL = 80
START_OF_SIGNAL_FROM_SHIFTED_X = -2.5

SURFACE_START = 980
SURFACE_END = 1010

WINDOW_HALF_WIDTH = 2.5
INTERPOLATION_POINTS = 201


# ============================================================
# feature extraction
# ============================================================

def extract_extension_block1_features(file_path):

    raw_measurements = load_measurement(file_path)

    x = raw_measurements["x"].to_numpy()
    y = raw_measurements["y"].to_numpy()

    if len(y) == 0:
        raise ValueError("empty measurement")

    # ========================================================
    # surface peak
    # ========================================================

    surface_mask = ((x >= SURFACE_START) & (x <= SURFACE_END))

    surface_indices = np.where(surface_mask)[0]

    if len(surface_indices) == 0:
        raise ValueError("surface region missing")

    y_centered = y - np.median(y)

    surface_idx = surface_indices[np.argmax(np.abs(y_centered[surface_mask]))]

    surface_time = x[surface_idx]

    x = x - surface_time

    # ========================================================
    # baseline / noise
    # ========================================================

    noise_region_mask = ((x >= NOISE_REGION_START) & (x <= END_OF_SIGNAL))

    if not np.any(noise_region_mask):
        raise ValueError("noise region missing")

    noise_region = np.mean(y[noise_region_mask])

    noise_std = np.std(y[noise_region_mask] - noise_region)

    if (
            not np.isfinite(noise_std) or noise_std == 0):
        raise ValueError("invalid noise std")

    surface_amp = (y[surface_idx] - noise_region)

    if (
            not np.isfinite(surface_amp) or surface_amp == 0):
        raise ValueError("invalid surface amp")

    # ========================================================
    # baseline corrected complete signal
    # ========================================================

    total_mask = ((x >= START_OF_SIGNAL_FROM_SHIFTED_X) & (x <= END_OF_SIGNAL))

    x_total = x[total_mask]

    y_total = (y[total_mask] - noise_region)

    # ========================================================
    # post peak region
    # ========================================================

    post_peak_mask = ((x >= POST_PEAK_START) & (x <= END_OF_SIGNAL))

    x = x[post_peak_mask]

    y = (y[post_peak_mask] - noise_region)

    if len(y) < 3:
        raise ValueError("post peak region too small")

    # ========================================================
    # second peak features
    # ========================================================

    second_max_peaks, second_max_properties = (find_peaks(y, prominence=0, width=0))

    second_min_peaks, second_min_properties = (find_peaks(-y, prominence=0, width=0))

    # positive maxima
    positive_mask = (y[second_max_peaks] > 0)

    positive_peaks = (second_max_peaks[positive_mask])

    positive_prominences = (second_max_properties["prominences"][positive_mask])

    positive_widths = (second_max_properties["widths"][positive_mask])

    # negative minima
    negative_mask = (y[second_min_peaks] < 0)

    negative_peaks = (second_min_peaks[negative_mask])

    negative_prominences = (second_min_properties["prominences"][negative_mask])

    negative_widths = (second_min_properties["widths"][negative_mask])

    if len(positive_peaks) == 0:
        raise ValueError("no positive post peak")

    if len(negative_peaks) == 0:
        raise ValueError("no negative post peak")

    # largest positive maximum
    second_peak_max_prop_idx = np.argmax(y[positive_peaks])

    second_peak_max_signal_idx = (positive_peaks[second_peak_max_prop_idx])

    second_peak_max = y[second_peak_max_signal_idx]

    second_peak_max_prominence = (positive_prominences[second_peak_max_prop_idx])

    second_peak_max_width = (positive_widths[second_peak_max_prop_idx])

    # smallest negative minimum
    second_peak_min_prop_idx = np.argmin(y[negative_peaks])

    second_peak_min_signal_idx = (negative_peaks[second_peak_min_prop_idx])

    second_peak_min = y[second_peak_min_signal_idx]

    second_peak_min_prominence = (negative_prominences[second_peak_min_prop_idx])

    second_peak_min_width = (negative_widths[second_peak_min_prop_idx])

    second_peak_distance = abs(x[second_peak_max_signal_idx] - x[second_peak_min_signal_idx])

    second_peak_max_to_min_ratio = abs(second_peak_max / second_peak_min)

    second_peak_amp = (second_peak_max
        if second_peak_max > abs(second_peak_min)
        else second_peak_min)

    second_peak_snr = (abs(second_peak_amp) / noise_std)

    second_peak_to_surface_ratio = (abs(second_peak_amp) / abs(surface_amp))

    # ========================================================
    # local std / energy
    # ========================================================

    y_fifths = np.array_split(y, 5)

    x_fifths = np.array_split(x, 5)

    fifths_stds = [np.std(window) for window in y_fifths]

    fifths_energies = [np.sum(window ** 2) for window in y_fifths]

    max_window_std_idx = np.argmax(fifths_stds)

    max_window_std = fifths_stds[max_window_std_idx]

    x_of_max_window_std = np.mean(x_fifths[max_window_std_idx])

    max_window_energy_idx = np.argmax(fifths_energies)

    max_window_energy = fifths_energies[max_window_energy_idx]

    x_of_max_window_energy = np.mean(x_fifths[max_window_energy_idx])

    # ========================================================
    # post peak stats
    # ========================================================

    post_peak_energy = np.sum(y ** 2)

    post_peak_std = np.std(y)

    post_peak_abs_area = np.trapz(np.abs(y), x)

    # ========================================================
    # derivatives
    # ========================================================

    dy_dx = np.gradient(y, x)

    post_peak_max_pos_slope = np.max(dy_dx)

    post_peak_max_neg_slope = np.min(dy_dx)

    mean_abs_derivative = np.mean(np.abs(dy_dx))

    std_derivative = np.std(dy_dx)

    dy2_dx2 = np.gradient(dy_dx,x)

    post_peak_max_sec_pos_slope = np.max(dy2_dx2)

    post_peak_max_sec_neg_slope = np.min(dy2_dx2)

    # ========================================================
    # surface / second peak correlation
    # ========================================================

    surface_mask = ((x_total >= -WINDOW_HALF_WIDTH) & (x_total <= WINDOW_HALF_WIDTH))

    surface_x = x_total[surface_mask]

    surface_y = y_total[surface_mask]

    if (second_peak_max > abs(second_peak_min)
    ):
        second_peak_signal_idx = (
            second_peak_max_signal_idx
        )
    else:
        second_peak_signal_idx = (
            second_peak_min_signal_idx
        )

    second_peak_position = x[second_peak_signal_idx]

    second_mask = ((x_total >= second_peak_position - WINDOW_HALF_WIDTH) & (x_total <= second_peak_position + WINDOW_HALF_WIDTH))

    second_x = (x_total[second_mask] - second_peak_position)

    second_y = y_total[second_mask]

    if len(surface_x) < 2:
        raise ValueError("surface window too small")

    if len(second_x) < 2:
        raise ValueError("second peak window too small")

    x_interp = np.linspace(-WINDOW_HALF_WIDTH, WINDOW_HALF_WIDTH, INTERPOLATION_POINTS)

    surface_interp = np.interp(x_interp, surface_x, surface_y)

    second_interp = np.interp(x_interp, second_x, second_y)

    if np.std(surface_interp) == 0:
        raise ValueError("constant surface window")

    if np.std(second_interp) == 0:
        raise ValueError("constant second peak window")

    second_peak_surface_corr = np.corrcoef(surface_interp, second_interp)[0, 1]

    if not np.isfinite(second_peak_surface_corr):
        raise ValueError("invalid shape correlation")

    # ========================================================
    # result
    # ========================================================

    return {
        # second peak
        "second_peak_max": second_peak_max,
        "second_peak_min": second_peak_min,
        "second_peak_max_prominence": second_peak_max_prominence,
        "second_peak_max_width": second_peak_max_width,
        "second_peak_min_prominence": second_peak_min_prominence,
        "second_peak_min_width": second_peak_min_width,
        "second_peak_distance": second_peak_distance,
        "second_peak_max_to_min_ratio": second_peak_max_to_min_ratio,
        "second_peak_amp": second_peak_amp,
        "second_peak_snr": second_peak_snr,
        "second_peak_to_surface_ratio": second_peak_to_surface_ratio,

        # local std / energy
        "first_fifth_std": fifths_stds[0],
        "second_fifth_std": fifths_stds[1],
        "third_fifth_std": fifths_stds[2],
        "fourth_fifth_std": fifths_stds[3],
        "fifth_fifth_std": fifths_stds[4],

        "max_window_std": max_window_std,
        "x_of_max_window_std": x_of_max_window_std,

        "first_fifth_energie": fifths_energies[0],
        "second_fifth_energie": fifths_energies[1],
        "third_fifth_energie": fifths_energies[2],
        "fourth_fifth_energie": fifths_energies[3],
        "fifth_fifth_energie": fifths_energies[4],

        "max_window_energy": max_window_energy,
        "x_of_max_window_energy": x_of_max_window_energy,

        # post peak
        "post_peak_energy": post_peak_energy,
        "post_peak_std": post_peak_std,
        "post_peak_abs_area": post_peak_abs_area,

        # derivatives
        "post_peak_max_pos_slope": post_peak_max_pos_slope,
        "post_peak_max_neg_slope": post_peak_max_neg_slope,
        "mean_abs_derivative": mean_abs_derivative,
        "std_derivative": std_derivative,
        "post_peak_max_sec_pos_slope": post_peak_max_sec_pos_slope,
        "post_peak_max_sec_neg_slope": post_peak_max_sec_neg_slope,

        # correlation
        "second_peak_surface_corr": second_peak_surface_corr,
    }
