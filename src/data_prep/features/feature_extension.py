from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences


# ============================================================
# Pfade anpassen
# ============================================================

INPUT_FEATURE_TABLE = Path(
    "/results/feature_tables/feature_baseline.csv"
)

MEASUREMENT_ROOT = Path(
    "/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/data"
)

OUTPUT_PATH = Path(
    "/results/feature_tables/feature_table_extension.csv"
)


# ============================================================
# Testmodus
# ============================================================

TEST_MODE = False
N_TEST = 5000

RANDOM_STATE = 42


# ============================================================
# Einstellungen für Feature-Berechnung
# ============================================================

# Falls du die Oberfläche grob einschränken möchtest.
# Wenn beide None sind, wird einfach der stärkste Ausschlag im gesamten Signal genommen.
SURFACE_X_MIN = None
SURFACE_X_MAX = None

# Bereich nach der Oberfläche, in dem ein auffälliger Post-Peak gesucht wird
POST_MIN_DELAY = 2.0
POST_MAX_DELAY = 80.0

# Block 2: hier ggf. an deine bisherige Blockdefinition anpassen
BLOCK2_X_MIN = 1020
BLOCK2_X_MAX = 1080

# Für Rauschschätzung: letzter Anteil des Signals
NOISE_TAIL_FRACTION = 0.10

# Peak-Erkennung
PEAK_DISTANCE_POINTS = 10
PEAK_PROMINENCE_NOISE_FACTOR = 2.0


# ============================================================
# Hilfsfunktionen
# ============================================================

def safe_div(a, b):
    if b is None or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def resolve_measurement_path(row):
    """
    Versucht, den Pfad zur Messdatei aus der Feature-Tabelle zu bestimmen.

    Unterstützt:
    - absolute file_path
    - relative_path relativ zu MEASUREMENT_ROOT
    - filename relativ zu MEASUREMENT_ROOT
    """

    if "file_path" in row and pd.notna(row["file_path"]):
        p = Path(str(row["file_path"]))
        if p.exists():
            return p

    if "relative_path" in row and pd.notna(row["relative_path"]):
        p = MEASUREMENT_ROOT / str(row["relative_path"])
        if p.exists():
            return p

    if "filename" in row and pd.notna(row["filename"]):
        p = MEASUREMENT_ROOT / str(row["filename"])
        if p.exists():
            return p

    return None


def read_measurement_file(path):
    """
    Liest eine Messdatei robust ein.

    Erwartung:
    - meistens zwei numerische Spalten: x und y
    - falls nur eine numerische Spalte vorhanden ist: x = 0, 1, 2, ...
    """

    try:
        raw = pd.read_csv(
            path,
            sep=r"\s+|,|;",
            comment="#",
            header=None,
            engine="python",
        )
    except Exception:
        return None, None

    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(axis=1, how="all")
    raw = raw.dropna(axis=0, how="all")

    if raw.shape[0] < 5:
        return None, None

    if raw.shape[1] >= 2:
        x = raw.iloc[:, 0].to_numpy(dtype=float)
        y = raw.iloc[:, 1].to_numpy(dtype=float)
    elif raw.shape[1] == 1:
        y = raw.iloc[:, 0].to_numpy(dtype=float)
        x = np.arange(len(y), dtype=float)
    else:
        return None, None

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 5:
        return None, None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    return x, y


def get_surface_index(x, y):
    """
    Oberfläche = stärkster Absolutausschlag.
    Optional eingeschränkt durch SURFACE_X_MIN / SURFACE_X_MAX.
    """

    mask = np.ones(len(x), dtype=bool)

    if SURFACE_X_MIN is not None:
        mask &= x >= SURFACE_X_MIN

    if SURFACE_X_MAX is not None:
        mask &= x <= SURFACE_X_MAX

    idx_candidates = np.where(mask)[0]

    if len(idx_candidates) == 0:
        return int(np.argmax(np.abs(y)))

    local_idx = np.argmax(np.abs(y[idx_candidates]))
    return int(idx_candidates[local_idx])


def estimate_noise_std(y):
    """
    Rauschschätzung über den letzten Teil des Signals.
    """

    n_tail = max(20, int(len(y) * NOISE_TAIL_FRACTION))
    n_tail = min(n_tail, len(y))

    tail = y[-n_tail:]

    if len(tail) < 2:
        return np.nan

    return float(np.std(tail))


def calculate_features_for_signal(x, y):
    """
    Berechnet neue Features für eine einzelne Messung.
    """

    result = {
        "surface_amp": np.nan,
        "surface_time": np.nan,
        "post_peak_amp": np.nan,
        "post_peak_time": np.nan,
        "post_peak_delay": np.nan,
        "post_peak_prominence": np.nan,
        "post_peak_to_surface_ratio": np.nan,
        "noise_std": np.nan,
        "post_peak_snr": np.nan,
        "block2_abs_max": np.nan,
        "block2_energy": np.nan,
        "block2_std": np.nan,
        "block2_to_total_energy_ratio": np.nan,
        "num_peaks_after_surface": np.nan,
        "max_abs_slope": np.nan,
    }

    if x is None or y is None or len(x) < 5:
        return result

    # ========================================================
    # Oberfläche
    # ========================================================

    surface_idx = get_surface_index(x, y)

    surface_amp = float(y[surface_idx])
    surface_time = float(x[surface_idx])

    result["surface_amp"] = surface_amp
    result["surface_time"] = surface_time

    # ========================================================
    # Rauschen
    # ========================================================

    noise_std = estimate_noise_std(y)
    result["noise_std"] = noise_std

    # ========================================================
    # Post-Peak nach Oberfläche
    # ========================================================

    post_mask = (
        (x >= surface_time + POST_MIN_DELAY)
        & (x <= surface_time + POST_MAX_DELAY)
    )

    x_post = x[post_mask]
    y_post = y[post_mask]

    if len(x_post) >= 5:
        abs_post = np.abs(y_post)

        if pd.notna(noise_std) and noise_std > 0:
            min_prominence = PEAK_PROMINENCE_NOISE_FACTOR * noise_std
        else:
            min_prominence = None

        peaks, properties = find_peaks(
            abs_post,
            distance=PEAK_DISTANCE_POINTS,
            prominence=min_prominence,
        )

        result["num_peaks_after_surface"] = int(len(peaks))

        if len(peaks) > 0:
            prominences = properties["prominences"]
            best_peak_local = peaks[int(np.argmax(prominences))]
            best_prominence = float(np.max(prominences))
        else:
            # Fallback: stärkster Punkt im Post-Bereich
            best_peak_local = int(np.argmax(abs_post))
            best_prominence = np.nan

        post_peak_amp = float(y_post[best_peak_local])
        post_peak_time = float(x_post[best_peak_local])
        post_peak_delay = post_peak_time - surface_time

        result["post_peak_amp"] = post_peak_amp
        result["post_peak_time"] = post_peak_time
        result["post_peak_delay"] = post_peak_delay
        result["post_peak_prominence"] = best_prominence
        result["post_peak_to_surface_ratio"] = safe_div(
            abs(post_peak_amp),
            abs(surface_amp),
        )
        result["post_peak_snr"] = safe_div(
            abs(post_peak_amp),
            noise_std,
        )

    # ========================================================
    # Block-2-Features
    # ========================================================

    block2_mask = (x >= BLOCK2_X_MIN) & (x <= BLOCK2_X_MAX)
    y_block2 = y[block2_mask]

    total_energy = float(np.sum(y ** 2))

    if len(y_block2) >= 2:
        block2_energy = float(np.sum(y_block2 ** 2))

        result["block2_abs_max"] = float(np.max(np.abs(y_block2)))
        result["block2_energy"] = block2_energy
        result["block2_std"] = float(np.std(y_block2))
        result["block2_to_total_energy_ratio"] = safe_div(
            block2_energy,
            total_energy,
        )

    # ========================================================
    # Steigungsfeature
    # ========================================================

    dx = np.diff(x)
    dy = np.diff(y)

    valid_dx = dx != 0

    if np.any(valid_dx):
        slopes = dy[valid_dx] / dx[valid_dx]
        result["max_abs_slope"] = float(np.max(np.abs(slopes)))

    return result


# ============================================================
# Hauptprogramm
# ============================================================

def main():
    df = pd.read_csv(INPUT_FEATURE_TABLE)

    print("Geladene Feature-Tabelle:")
    print(df.shape)

    if TEST_MODE:
        n = min(N_TEST, len(df))
        df_work = df.head(n).copy()
        print(f"\nTEST_MODE aktiv: Berechne nur die ersten {n} Samples.")
    else:
        df_work = df.copy()
        print(f"\nTEST_MODE aus: Berechne alle {len(df_work)} Samples.")

    new_feature_rows = []
    missing_files = 0
    failed_files = 0

    for i, (_, row) in enumerate(df_work.iterrows(), start=1):
        path = resolve_measurement_path(row)

        if path is None:
            missing_files += 1
            new_feature_rows.append(calculate_features_for_signal(None, None))
            continue

        x, y = read_measurement_file(path)

        if x is None or y is None:
            failed_files += 1
            new_feature_rows.append(calculate_features_for_signal(None, None))
            continue

        features = calculate_features_for_signal(x, y)
        new_feature_rows.append(features)

        if i % 500 == 0:
            print(f"{i} / {len(df_work)} Messungen verarbeitet")

    new_features_df = pd.DataFrame(new_feature_rows)

    df_out = pd.concat(
        [
            df_work.reset_index(drop=True),
            new_features_df.reset_index(drop=True),
        ],
        axis=1,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print("\nFertig.")
    print(f"Gespeichert unter: {OUTPUT_PATH}")
    print(f"Zeilen: {len(df_out)}")
    print(f"Neue Features: {new_features_df.shape[1]}")
    print(f"Fehlende Dateien: {missing_files}")
    print(f"Nicht lesbare Dateien: {failed_files}")

    print("\nNeue Feature-Spalten:")
    for col in new_features_df.columns:
        print("-", col)

    print("\nVorschau:")
    print(df_out.head())


if __name__ == "__main__":
    main()