# Random Forest Ergebnisse

| Approach | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Kritische FN | FN-Rate | False Alarms | FA-Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.8630 | 0.6881 | 0.7165 | 0.8481 | 685 | 29.30% | 130 | 1.90% |
| Merge: True, Thresholds: False | 0.8672 | 0.7951 | 0.8170 | 0.8626 | 628 | 26.86% | 184 | 2.69% |
| Merge: False, Thresholds: True | 0.7480 | 0.7634 | 0.6799 | 0.7794 | 116 | 4.96% | 1785 | 26.10% |
| Merge: True, Thresholds: True | 0.7595 | 0.7565 | 0.7330 | 0.7776 | 128 | 5.47% | 1761 | 25.75% |
| EXTENDED: M False, T False | 0.9022 | 0.7676 | 0.8021 | 0.8935 | 547 | 23.40% | 130 | 1.90% |
| EXTENDED: M True, T False | 0.9094 | 0.8617 | 0.8862 | 0.9073 | 501 | 21.43% | 149 | 2.18% |
| EXTENDED: M False, T True | 0.8113 | 0.8353 | 0.7556 | 0.8353 | 90 | 3.85% | 1448 | 21.18% |
| EXTENDED: M True, T True | 0.8188 | 0.8344 | 0.8098 | 0.8318 | 84 | 3.59% | 1466 | 21.44% || EXTENDED: M True, T False | 0.8672 | 0.7951 | 0.8170 | 0.8626 | 628 | 26.86% | 184 | 2.69% |