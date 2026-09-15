import pandas as pd
from matplotlib import pyplot as plt

from src.paths import FEATURE_BASELINE_ROOT, BASELINE_ANALYSIS_RESULTS

from pathlib import Path

# ============================================================
# daten laden, features bestimmen
# ============================================================

INPUT_PATH = FEATURE_BASELINE_ROOT

OUTPUT_DIR = BASELINE_ANALYSIS_RESULTS
OUTPUT_PATH = OUTPUT_DIR / "stats_and_correlation_baseline_features.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEATMAP_NAME = "baseline_features_heatmap"
PLOT_NAME = "Korrelationsmatrix der Baseline-Features"

df = pd.read_csv(INPUT_PATH)

EXCLUDE_COLUMNES = ["label", "file_path", "x", "y", "source_dataset", "instance_id"]

FEATURES = [col for col in df.columns if col not in EXCLUDE_COLUMNES]

corr = df[FEATURES].corr()

# ============================================================
# heatmap
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

plt.title(f"{PLOT_NAME}")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / f"{HEATMAP_NAME}.png",
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


corr_matrix = df[FEATURES].corr()

high_corr = []

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):

        corr = corr_matrix.iloc[i, j]

        if abs(corr) >= 0.8:
            high_corr.append({
                "feature_1": corr_matrix.columns[i],
                "feature_2": corr_matrix.columns[j],
                "correlation": corr
            })

high_corr_df = pd.DataFrame(high_corr)

high_corr_df["abs_correlation"] = high_corr_df["correlation"].abs()

high_corr_df = high_corr_df.sort_values(
    "abs_correlation",
    ascending=False
)

print(high_corr_df.to_string(index=False))
