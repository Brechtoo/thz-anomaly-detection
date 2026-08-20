from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import FEATURE_BASELINE_ROOT

from scipy.signal import find_peaks

"""
erster neuer feature-block

-> max_prominence, max_width, min_prominence, min_width
-> 5 x window std, max_window_std, x_of_max_window_std
-> post_peak_energy, post_peak_abs_area, post_peak_std
-> second_peak_min, second_peak_max, second_peak_amp, second_peak_position, second_peak_peak_distance
-> second_peak_prominence, second_peak_snr, second_peak_max_to_min_ratio, second_peak_abs_min_to_max_ratio,
-> second_peak_to_surface_ratio
-> post_max_pos_slope, post_max_neg_slope, mean_abs_derivative, max_sec_abs_derivative, std_derivative
"""

# ============================================================
# settings
# ============================================================

INPUT_PATH = FEATURE_BASELINE_ROOT

OUTPUT_DIR = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised"
                   "/results/feature_tables")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "feature_extension_block1.csv"

POST_PEAK_START = 3
NOISE_REGION_START = 70
END_OF_SIGNAL = 80
START_OF_SIGNAL_FROM_SHIFTED_X = -2.5

# ============================================================
# load data
# ============================================================

df = pd.read_csv(INPUT_PATH)

# ============================================================
# feature extension extraction
# ============================================================

def extract_feature_extension_block1(file_path):

    raw_measurements = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        comment="#",
        names=["x", "y"],
        usecols=[0, 1],
    )

    # umwandlung in zahlen
    raw_measurements["x"] = pd.to_numeric(raw_measurements["x"], errors="coerce")
    raw_measurements["y"] = pd.to_numeric(raw_measurements["y"], errors="coerce")
    raw_measurements = raw_measurements.dropna()  # fehlende werte droppen

    x = raw_measurements["x"].to_numpy()
    y = raw_measurements["y"].to_numpy()

    if len(y) == 0:
        raise ValueError("empty measurement")


    # ============================================================
    # surface peak
    # ============================================================

    SURFACE_START = 980
    SURFACE_END = 1010

    surface_mask = ((x >= SURFACE_START) & (x <= SURFACE_END))

    surface_indices = np.where(surface_mask)[0]

    y_centered = y - np.median(y)

    surface_idx = surface_indices[np.argmax(np.abs(y_centered[surface_mask]))]

    surface_time = x[surface_idx]
    x = x - surface_time

    # ============================================================
    # zeitfenster und baseline
    # ============================================================

    noise_region_mask = (x >= NOISE_REGION_START) & (x <= END_OF_SIGNAL)

    if not np.any(noise_region_mask):
        raise ValueError("noise region missing")

    noise_region = np.mean(y[noise_region_mask])
    noise_std = np.std(y[noise_region_mask] - noise_region)

    if not np.isfinite(noise_std) or noise_std == 0:
        raise ValueError("invalid noise std")

    surface_amp = y[surface_idx] - noise_region

    if not np.isfinite(surface_amp) or surface_amp == 0:
        raise ValueError("invalid surface amp")

    # baseline korrigiertes gesamtsignal
    baseline_corrected_signal_mask = (x >= START_OF_SIGNAL_FROM_SHIFTED_X) & (x <= END_OF_SIGNAL)

    x_total = x[baseline_corrected_signal_mask]
    y_total = y[baseline_corrected_signal_mask] - noise_region

    # für die meiste feature berechnung
    post_peak_mask = (x >= POST_PEAK_START) & (x <= END_OF_SIGNAL)
    
    x = x[post_peak_mask]
    y = y[post_peak_mask] - noise_region

    if len(y) < 3:
        raise ValueError("post peak region too small")

    # ============================================================
    # second peak merkmale (11):
    # second_peak_max, second_peak_min, second_peak_max_prominence, second_peak_min_prominence, second_peak_max_width, second_peak_min_width
    # second_peak_peak_distance, second_peak_snr, second_peak_amp,
    # second_peak_max_to_min_ratio, second_peak_min_to_max_ratio, second_peak_to_surface_ratio
    # ============================================================

    second_max_peaks, second_max_properties = find_peaks(
        y,
        prominence=0,
        width=0
    )

    second_min_peaks, second_min_properties = find_peaks(
        -y,
        prominence=0,
        width=0
    )

    # nur positive Maxima
    positive_mask = y[second_max_peaks] > 0
    positive_peaks = second_max_peaks[positive_mask]
    positive_prominences = second_max_properties["prominences"][positive_mask]
    positive_widths = second_max_properties["widths"][positive_mask]

    # nur negative Minima
    negative_mask = y[second_min_peaks] < 0
    negative_peaks = second_min_peaks[negative_mask]
    negative_prominences = second_min_properties["prominences"][negative_mask]
    negative_widths = second_min_properties["widths"][negative_mask]

    if len(positive_peaks) == 0:
        raise ValueError("no positive post peak")

    if len(negative_peaks) == 0:
        raise ValueError("no negative post peak")

    # größtes positives Maximum
    second_peak_max_prop_idx = np.argmax(y[positive_peaks])
    second_peak_max_signal_idx = positive_peaks[second_peak_max_prop_idx]

    second_peak_max = y[second_peak_max_signal_idx]
    second_peak_max_prominence = positive_prominences[second_peak_max_prop_idx]
    second_peak_max_width = positive_widths[second_peak_max_prop_idx]

    # kleinstes negatives Minimum
    second_peak_min_prop_idx = np.argmin(y[negative_peaks])
    second_peak_min_signal_idx = negative_peaks[second_peak_min_prop_idx]

    second_peak_min = y[second_peak_min_signal_idx]
    second_peak_min_prominence = negative_prominences[second_peak_min_prop_idx]
    second_peak_min_width = negative_widths[second_peak_min_prop_idx]

    # zeitlicher abstand zwischen pos und neg peak
    second_peak_distance = abs(x[second_peak_max_signal_idx] - x[second_peak_min_signal_idx])

    # verhältnis der amplituden
    second_peak_max_to_min_ratio = abs(second_peak_max / second_peak_min)

    # größter peak
    second_peak_amp = (second_peak_max if second_peak_max > abs(second_peak_min)else second_peak_min)

    second_peak_snr = abs(second_peak_amp) / noise_std

    second_peak_to_surface_ratio = (abs(second_peak_amp) / abs(surface_amp))

    # ============================================================
    # lokale streuung und energie (14):
    # 5 x window std, max_window_std, x_of_max_window_std
    # 5 x window std, max_window_std, x_of_max_window_std
    # ============================================================

    y_fifths = np.array_split(y, 5)
    x_fifths = np.array_split(x, 5)

    fifths_stds = [np.std(window) for window in y_fifths]
    fifths_energies = [np.sum(window**2) for window in y_fifths]

    max_window_std_idx = np.argmax(fifths_stds)
    max_window_std = fifths_stds[max_window_std_idx]
    x_of_max_window_std = np.mean(x_fifths[max_window_std_idx])

    max_window_energies_idx = np.argmax(fifths_energies)
    max_window_energy = fifths_energies[max_window_energies_idx]
    x_of_max_window_energy = np.mean(x_fifths[max_window_energies_idx])

    # ============================================================
    # post peak stats (3): post_peak_energy, post_peak_abs_area, post_peak_std
    # ============================================================

    post_peak_energy = np.sum(y**2)

    post_peak_std = np.std(y)

    post_peak_abs_area = np.trapz(np.abs(y), x)

    # ============================================================
    # signaländerungs- und ableitungsmerkmale (6):
    # post_peak_max_pos_slope, post_peak_max_neg_slope, post_peak_max_sec_pos_slope, post_peak_max_sec_neg_slope
    # mean_abs_derivative, std_derivative
    # ============================================================

    dy_dx = np.gradient(y, x)

    post_peak_max_pos_slope = np.max(dy_dx)
    post_peak_max_neg_slope = np.min(dy_dx)

    mean_abs_derivative = np.mean(np.abs(dy_dx))
    std_derivative = np.std(dy_dx)

    dy2_dx2 = np.gradient(dy_dx, x)

    post_peak_max_sec_pos_slope = np.max(dy2_dx2)
    post_peak_max_sec_neg_slope = np.min(dy2_dx2)

    # ============================================================
    # korrelation von peak und surface (1): shape_corr
    # ============================================================

    WINDOW_HALF_WIDTH = 2.5
    INTERPOLATION_POINTS = 201

    # Surface-Fenster
    surface_mask = ((x_total >= -WINDOW_HALF_WIDTH) & (x_total <= WINDOW_HALF_WIDTH))

    surface_x = x_total[surface_mask]
    surface_y = y_total[surface_mask]

    if second_peak_max > abs(second_peak_min):
        second_peak_signal_idx = second_peak_max_signal_idx
    else:
        second_peak_signal_idx = second_peak_min_signal_idx

    second_peak_position = x[second_peak_signal_idx]

    second_mask = ((x_total >= second_peak_position - WINDOW_HALF_WIDTH) & (x_total <= second_peak_position + WINDOW_HALF_WIDTH))

    second_x = x_total[second_mask] - second_peak_position
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

    feature_extension_block1_dict = {
        # A1. second peak merkmale
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

        # B1. lokale streuung und energie
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

        # C1. post peak stats
        "post_peak_energy": post_peak_energy,
        "post_peak_std": post_peak_std,
        "post_peak_abs_area": post_peak_abs_area,

        # D1. signaländerungs- und ableitungsmerkmale
        "post_peak_max_pos_slope": post_peak_max_pos_slope,
        "post_peak_max_neg_slope": post_peak_max_neg_slope,
        "mean_abs_derivative": mean_abs_derivative,
        "std_derivative": std_derivative,
        "post_peak_max_sec_pos_slope": post_peak_max_sec_pos_slope,
        "post_peak_max_sec_neg_slope": post_peak_max_sec_neg_slope,

        # E1. formkorrelation
        "second_peak_surface_corr": second_peak_surface_corr
    }

    return feature_extension_block1_dict

rows = []
error_rows = []

for i, row in df.iterrows():

    file_path = row["file_path"]

    try:
        features = extract_feature_extension_block1(file_path)
        features["file_path"] = file_path
        rows.append(features)

    except Exception as e:
        print(f"FEHLER bei: {file_path}")
        print(f"  {type(e).__name__}: {e}")

        error_rows.append({
            "file_path": file_path,
            "error_type": type(e).__name__,
            "error_message": str(e)
        })

        rows.append({
            "file_path": file_path
        })

    if i % 1000 == 0:
        print(f"{i} messungen verarbeitet...")


# ============================================================
# save
# ============================================================

new_features = pd.DataFrame(rows)

feature_df = df.merge(new_features,on="file_path",how="left")
feature_df.to_csv(OUTPUT_PATH, index=False)

print("verarbeitete messungen:", len(feature_df))
print("features gespeichert unter:", OUTPUT_PATH)
