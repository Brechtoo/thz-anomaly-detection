import pandas as pd
import numpy as np

# ============================================================
# erste features erstellen
# ============================================================


INPUT_PATH = ("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised"
            "/results/measurements/combined_dataset.csv")
OUTPUT_PATH = ("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised"
               "/results/feature_tables/feature_baseline.csv")

# None = alle messungen verwenden
# n = n messungen verwenden
MAX_MEASUREMENTS = None

df = pd.read_csv(INPUT_PATH)

# zufälliges ziehen
if MAX_MEASUREMENTS is not None:
    df = df.sample(MAX_MEASUREMENTS, random_state=42)

# standardisierung
def extract_features(file_path):
    raw = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        comment="#",
        names=["x", "y"],
        usecols=[0, 1],
    )

    # umwandlung in zahlen
    raw["x"] = pd.to_numeric(raw["x"], errors="coerce")
    raw["y"] = pd.to_numeric(raw["y"], errors="coerce")
    raw = raw.dropna() # fehlende werte droppen

    x = raw["x"].to_numpy()
    y = raw["y"].to_numpy()

    if len(y) == 0:
        return {}

    abs_y = np.abs(y) # absolutbetrag jedes signalwertes
    thirds = np.array_split(y, 3)

    features = {
        # A. extremwerte
        "y_max": np.max(y),
        "y_min": np.min(y),
        "y_peak_to_peak": np.max(y) - np.min(y),

        # B. lage und streuung
        "y_mean": np.mean(y),
        "y_median": np.median(y),
        "y_std": np.std(y),
        "y_abs_mean": np.mean(abs_y),

        # C. energie und fläche
        "y_energy": np.sum(y ** 2),
        "y_abs_area": np.trapz(abs_y, x),

        # D. position des stärksten ausschlags
        "x_at_y_max": x[np.argmax(y)],
        "x_at_y_min": x[np.argmin(y)],

        # E. energie der drei signalbereiche
        "first_third_energy": np.sum(thirds[0] ** 2),
        "middle_third_energy": np.sum(thirds[1] ** 2),
        "last_third_energy": np.sum(thirds[2] ** 2),

        # F. standardabweichung der signalbereiche
        "first_third_std": np.std(thirds[0]),
        "middle_third_std": np.std(thirds[1]),
        "last_third_std": np.std(thirds[2]),
    }

    return features

rows = []


for i, row in df.iterrows():
    features = extract_features(row["file_path"])

    features["label"] = row["label"]
    features["file_path"] = row["file_path"]

    if "x" in df.columns:
        features["x"] = row["x"]
    if "y" in df.columns:
        features["y"] = row["y"]
    if "source_dataset" in df.columns:
        features["source_dataset"] = row["source_dataset"]
    if "instance_id" in df.columns:
        features["instance_id"] = row["instance_id"]

    rows.append(features)

    if i % 1000 == 0:
        print(f"{i} Messungen verarbeitet...")


feature_df = pd.DataFrame(rows)
feature_df.to_csv(OUTPUT_PATH, index=False)

print("Fertig.")
print("Verarbeitete Messungen:", len(feature_df))
print("Gespeichert unter:", OUTPUT_PATH)
print(feature_df.head())
print(feature_df.shape)
