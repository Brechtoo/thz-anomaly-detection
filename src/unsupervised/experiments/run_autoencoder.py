from pathlib import Path

import pandas as pd

from src.paths import FEATURE_EXTENSION_BLOCK1_ROOT, AUTOENCODER_RESULTS
from src.unsupervised.processing import evaluation, prepare_data
from src.unsupervised.models.autoencoder import train_autoencoder, calculate_reconstruction_scores


# ============================================================
# settings
# ============================================================

DATA_PATH = FEATURE_EXTENSION_BLOCK1_ROOT
OUTPUT_DIR = AUTOENCODER_RESULTS

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "label"

EXCLUDE_COLUMNS = [
    "label",
    "file_path",
    "x",
    "y",
    "source_dataset",
    "instance_id",
]

SPLIT_RANDOM_STATE = 42

MODEL_RANDOM_STATES = [1, 21, 42, 84, 123]

TOP_K_VALUES = [20, 50, 100, 500]


# ============================================================
# einzelne random states
# ============================================================

def run_single_random_state(X_train, X_test, y_test, random_state):
    model = train_autoencoder(X_train, random_state)

    scores = calculate_reconstruction_scores(model, X_test)

    metrics = evaluation.evaluate_anomaly_scores(y_test, scores, TOP_K_VALUES)

    return metrics


# ============================================================
# main
# ============================================================

def main():
    X, y = prepare_data.load_feature_data(DATA_PATH, TARGET_COL, EXCLUDE_COLUMNS)

    X_train, X_test, y_train, y_test = prepare_data.prepare_feature_data(X, y, random_state=SPLIT_RANDOM_STATE)

    all_results = []

    for random_state in MODEL_RANDOM_STATES:
        metrics = run_single_random_state(X_train, X_test, y_test, random_state)

        metrics["random_state"] = random_state

        all_results.append(metrics)

    results_df = pd.DataFrame(all_results)

    columns = ["random_state"] + [col for col in results_df.columns if col != "random_state"]

    results_df = results_df[columns]

    results_df.to_csv(OUTPUT_DIR / "autoencoder_results.csv", index=False)

    metric_columns = [col for col in results_df.columns if col != "random_state"]

    summary = results_df[metric_columns].agg(["mean", "std", "min", "max"]).T

    summary.to_csv(OUTPUT_DIR / "autoencoder_summary.csv")


if __name__ == "__main__":
    main()