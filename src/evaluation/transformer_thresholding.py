from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.thresholding import threshold_predict
from src.evaluation.metrics import (
    evaluate_predictions,
    calculate_error_counts,
)


PREDICTIONS_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/transformer/transformer_unweighted_validation_predictions.csv")

GREEN_THRESHOLDS = [
    0.50, 0.55, 0.60, 0.65, 0.70
]

RED_THRESHOLDS = [round(0.20 + i * 0.05, 2) for i in range(11)]

df = pd.read_csv(PREDICTIONS_PATH)

y_val = df["label"]

y_val_rgy = y_val.replace({0: 2})

probabilities = df[
    ["proba_0", "proba_1", "proba_2", "proba_3"]
].to_numpy()

classes = np.array([0, 1, 2, 3])

results = []

for green_threshold in GREEN_THRESHOLDS:
    for red_threshold in RED_THRESHOLDS:

        y_pred = threshold_predict(
            probabilities,
            classes,
            red_threshold=red_threshold,
            green_threshold=green_threshold,
        )

        metrics = evaluate_predictions(
            y_val_rgy,
            y_pred
        )

        errors = calculate_error_counts(
            y_val,
            y_pred
        )

        results.append({
            "red_threshold": red_threshold,
            "green_threshold": green_threshold,
            **metrics,
            **errors,
        })

results_df = pd.DataFrame(results)

candidates = results_df[
    (results_df["green_to_yellow_rate"] <= 15)
    & (results_df["false_alarm_rate"] <= 5)
]

print(
    candidates.sort_values(
        ["critical_errors", "macro_f1"],
        ascending=[True, False]
    ).head(20).to_string(index=False)
)