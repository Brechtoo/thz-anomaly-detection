import pandas as pd
from src.paths import FEATURE_BASELINE_ROOT, BASELINE_ANALYSIS_RESULTS

from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# daten laden, features aufgrund der ergebnisse von
# basic_stats_and_correlations.py bestimmen
# ============================================================

INPUT_PATH = FEATURE_BASELINE_ROOT

OUTPUT_DIR = BASELINE_ANALYSIS_RESULTS

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

FEATURES = [
    "y_min",
    "y_peak_to_peak",
    "y_std",
    "y_median",
    "first_third_std"
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


print(f"boxplots gespeichert unter: {OUTPUT_DIR}")