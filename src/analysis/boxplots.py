import pandas as pd
from src.paths import FEATURE_BASELINE_ROOT

from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# daten laden, features aufgrund der ergebnisse von
# basic_stats_and_correlations.py bestimmen
# ============================================================

INPUT_PATH = FEATURE_BASELINE_ROOT

OUTPUT_DIR = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/analysis/feature_baseline_analysis/stats_and_boxplots")
OUTPUT_PATH = OUTPUT_DIR / "boxplots.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FEATURE_BASELINE_ROOT)

FEATURES = [
    "y_min",
    "y_peak_to_peak",
    "first_third_std",
    "y_energy",
]

# ============================================================
# boxplots erzeugen
# ============================================================

for feature in FEATURES:

    plt.figure(figsize=(7, 5))

    df.boxplot(
        column=feature,
        by="label",
        grid=False,
    )

    plt.title(feature)
    plt.suptitle("")
    plt.xlabel("Klasse")
    plt.ylabel(feature)

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{feature}_boxplot.png"
    plt.savefig(output_path, dpi=300)

    plt.close()


print(f"Boxplots gespeichert unter: {OUTPUT_DIR}")