# HistGradientBoosting Ergebnisse

| Experiment | Prediction | P3 | P0 | Iterationen | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Kritische FN | FN-Rate | False Alarms | FA-Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hgb_baseline | Raw Argmax | - | - | 512 | 0.8201 | 0.8131 | 0.7540 | 0.8387 | 200 | 8.55% | 1025 | 14.99% |
| hgb_baseline | Thresholds | 0.80 | 0.60 | 512 | 0.7138 | 0.7977 | 0.6892 | 0.7570 | 61 | 2.61% | 2338 | 34.19% |
| hgb_extended | Raw Argmax | - | - | 488 | 0.8866 | 0.8775 | 0.8334 | 0.8950 | 170 | 7.27% | 648 | 9.48% |
| hgb_extended | Thresholds | 0.80 | 0.60 | 488 | 0.8027 | 0.8658 | 0.7703 | 0.8334 | 63 | 2.69% | 1673 | 24.47% |