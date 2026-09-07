from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def create_evaluation_dir(experiment_dir, model_name):
    evaluation_dir = Path(experiment_dir) / "evaluations" / model_name
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    return evaluation_dir


def save_confusion_matrix(y_true, y_pred, evaluation_dir):
    labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(cm, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])

    cm_df.to_csv(Path(evaluation_dir) / "confusion_matrix.csv")


def save_classification_report(y_true, y_pred, evaluation_dir):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

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
