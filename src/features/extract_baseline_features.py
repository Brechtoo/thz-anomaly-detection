import numpy as np

from src.features.loading import load_measurement


def extract_baseline_features(file_path):

    raw = load_measurement(file_path)

    x = raw["x"].to_numpy()
    y = raw["y"].to_numpy()

    if len(y) == 0: raise ValueError("leere messsung")

    abs_y = np.abs(y)
    thirds = np.array_split(y, 3)

    features = {
        # A. Extremwerte
        "y_max": np.max(y),
        "y_min": np.min(y),
        "y_peak_to_peak": np.max(y) - np.min(y),

        # B. Lage und Streuung
        "y_mean": np.mean(y),
        "y_median": np.median(y),
        "y_std": np.std(y),
        "y_abs_mean": np.mean(abs_y),

        # C. Energie und Fläche
        "y_energy": np.sum(y ** 2),
        "y_abs_area": np.trapz(abs_y, x),

        # D. Position der Extremwerte
        "x_at_y_max": x[np.argmax(y)],
        "x_at_y_min": x[np.argmin(y)],

        # E. Energie der drei Signalbereiche
        "first_third_energy": np.sum(thirds[0] ** 2),
        "middle_third_energy": np.sum(thirds[1] ** 2),
        "last_third_energy": np.sum(thirds[2] ** 2),

        # F. Standardabweichung der Signalbereiche
        "first_third_std": np.std(thirds[0]),
        "middle_third_std": np.std(thirds[1]),
        "last_third_std": np.std(thirds[2]),
    }

    return features