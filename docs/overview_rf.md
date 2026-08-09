# Random Forest Ergebnisse

| Approach | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Kritische FN | FN-Rate | False Alarms | FA-Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.8630 | 0.6881 | 0.7165 | 0.8481 | 685 | 29.30% | 130 | 1.90% |
| Merge: True, Thresholds: False | 0.8672 | 0.7951 | 0.8170 | 0.8626 | 628 | 26.86% | 184 | 2.69% |
| Merge: False, Thresholds: True | 0.7480 | 0.7634 | 0.6799 | 0.7794 | 116 | 4.96% | 1785 | 26.10% |
| Merge: True, Thresholds: True | 0.7595 | 0.7565 | 0.7330 | 0.7776 | 128 | 5.47% | 1761 | 25.75% |
