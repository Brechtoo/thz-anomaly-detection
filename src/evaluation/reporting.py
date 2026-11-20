import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.evaluation.metrics import (
    calculate_error_counts,
    critical_breakdown,
    evaluate_predictions,
)


def create_evaluation_dir(experiment_dir, model_name):
    evaluation_dir = Path(experiment_dir) / "evaluations" / model_name
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    return evaluation_dir


def save_confusion_matrix(y_true, y_pred, evaluation_dir, *, labels=None):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(cm, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])

    cm_df.to_csv(Path(evaluation_dir) / "confusion_matrix.csv")


def save_classification_report(y_true, y_pred, evaluation_dir, *, labels=None):
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(Path(evaluation_dir) / "classification_report.csv")


def save_predictions(y_true, y_pred, evaluation_dir):
    predictions_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

    predictions_df["correct"] = (predictions_df["y_true"] == predictions_df["y_pred"])

    predictions_df.to_csv(Path(evaluation_dir) / "predictions.csv", index=False)


def save_evaluation_report(y_true, y_pred, experiment_dir, model_name):

    evaluation_dir = create_evaluation_dir(experiment_dir, model_name)

    save_confusion_matrix(y_true, y_pred, evaluation_dir)

    save_classification_report(y_true, y_pred, evaluation_dir)

    save_predictions(y_true, y_pred, evaluation_dir)


def metrics_for_group(truth, prediction):
    """Reuse four-label error definitions; add fixed-label recall/support.

    A zero error rate with zero denominator is the legacy convention, not
    evidence of safety. Explicit supports make such group results identifiable.
    """
    result = evaluate_predictions(truth, prediction)
    result = {k: float(v) for k, v in result.items()
              if k not in ('confusion_matrix', 'classification_report')}
    result['macro_f1'] = f1_score(truth, prediction, labels=[0, 1, 2, 3], average='macro', zero_division=0)
    result.update(calculate_error_counts(truth, prediction))
    report = classification_report(truth, prediction, labels=[0, 1, 2, 3], output_dict=True, zero_division=0)
    for label in range(4):
        result[f'recall_{label}'] = report[str(label)]['recall']
        result[f'support_{label}'] = report[str(label)]['support']
    result['critical_denominator'] = int(np.isin(truth, [1, 2]).sum())
    result['false_alarm_denominator'] = int(np.sum(np.asarray(truth) == 3))
    return result


def evaluate_holdout(metadata, prediction, output, *, include_critical_breakdown=False):
    """Persist predictions with group keys, full reports and subgroup metrics."""
    output.mkdir(parents=True, exist_ok=False)
    predictions = metadata[['instance_id', 'specimen_id', 'source_dataset', 'label']].copy()
    predictions['prediction'] = prediction
    predictions.to_csv(output / 'predictions.csv', index=False)
    truth = predictions.label.to_numpy()
    metrics = metrics_for_group(truth, prediction)
    if include_critical_breakdown:
        metrics.update(critical_breakdown(truth, prediction))
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    save_confusion_matrix(truth, prediction, output, labels=[0, 1, 2, 3])
    save_classification_report(truth, prediction, output, labels=[0, 1, 2, 3])
    for key in ('specimen_id', 'source_dataset'):
        rows = []
        for group, frame in predictions.groupby(key, sort=True):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # absent classes are explicit in supports
                rows.append({key: group, 'points': len(frame), **metrics_for_group(frame.label, frame.prediction)})
        pd.DataFrame(rows).to_csv(output / f'by_{key}.csv', index=False)
    return metrics
