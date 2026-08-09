# HistGradientBoosting Ergebnisse

| Experiment | Prediction | P3 | P0 | Iterationen | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Kritische FN | FN-Rate | False Alarms | FA-Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hgb_baseline | Raw Argmax | - | - | 512 | 0.8201 | 0.8131 | 0.7540 | 0.8387 | 200 | 8.55% | 1025 | 14.99% |
| hgb_baseline | Thresholds | 0.80 | 0.60 | 512 | 0.7138 | 0.7977 | 0.6892 | 0.7570 | 61 | 2.61% | 2338 | 34.19% |
