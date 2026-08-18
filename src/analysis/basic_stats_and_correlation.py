import pandas as pd
from matplotlib import pyplot as plt

from src.paths import FEATURE_BASELINE_ROOT

from pathlib import Path

# ============================================================
# daten laden, features bestimmen
# ============================================================

INPUT_PATH = FEATURE_BASELINE_ROOT

OUTPUT_DIR = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/analysis/feature_baseline_analysis/stats_and_boxplots")
OUTPUT_PATH = OUTPUT_DIR / "stats_and_correlation.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FEATURE_BASELINE_ROOT)

EXCLUDE_COLUMNES = ["label", "file_path", "x", "y", "source_dataset", "instance_id"]

FEATURES = [col for col in df.columns if col not in EXCLUDE_COLUMNES]

corr = df[FEATURES].corr()



# ============================================================
# Heatmap
# ============================================================

fig, ax = plt.subplots(figsize=(12, 10))

heatmap = ax.imshow(
    corr,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
)

ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)

ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)

fig.colorbar(
    heatmap,
    ax=ax,
    label="Pearson-Korrelation"
)

plt.title("Korrelationsmatrix der Baseline-Features")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

with open(OUTPUT_PATH, "w") as f:
    for feature in FEATURES:
        stats = df.groupby("label")[feature].agg([
            "mean",
            "median",
            "std"
        ])
        f.write(f"\n{feature}")
        f.write(stats.to_string())
    f.write("\n\n" + corr.to_string())
