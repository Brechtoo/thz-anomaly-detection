from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import numpy as np

def evaluate_predictions(y_true, y_pred):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, digits=4)
    }

    return results


def calculate_error_counts(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # ============================================================
    # criticals: actual 1, predicted 3
    # ============================================================

    critical_errors = np.sum((y_true == 1) & (y_pred == 3))

    critical_total = np.sum(y_true == 1)

    critical_error_rate = (critical_errors / critical_total * 100
        if critical_total > 0
        else 0.0
    )

    # ============================================================
    # false alarms: actual 3, predicted 1
    # ============================================================

    false_alarms = np.sum((y_true == 3) & (y_pred == 1))

    false_alarm_total = np.sum(y_true == 3)

    false_alarm_rate = (false_alarms / false_alarm_total * 100
        if false_alarm_total > 0
        else 0.0
    )

    # ============================================================
    # rot -> gelb: actual 1, predicted 0 oder 2
    # ============================================================

    red_to_yellow = np.sum((y_true == 1) & np.isin(y_pred, [0, 2]))

    red_to_yellow_rate = (red_to_yellow / critical_total * 100
        if critical_total > 0
        else 0.0
    )

    # ============================================================
    # grün -> gelb: actual 3, predicted 0 oder 2
    # ============================================================

    green_to_yellow = np.sum((y_true == 3) & np.isin(y_pred, [0, 2]))

    green_to_yellow_rate = (green_to_yellow / false_alarm_total * 100
        if false_alarm_total > 0
        else 0.0
    )

    return {
        "critical_errors": int(critical_errors),
        "critical_error_rate": critical_error_rate,

        "false_alarms": int(false_alarms),
        "false_alarm_rate": false_alarm_rate,

        "red_to_yellow": int(red_to_yellow),
        "red_to_yellow_rate": red_to_yellow_rate,

        "green_to_yellow": int(green_to_yellow),
        "green_to_yellow_rate": green_to_yellow_rate,
    }
