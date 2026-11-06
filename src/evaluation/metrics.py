import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ERROR_METRICS_VERSION = "four_label_v2"

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
    """Count errors from original four-class labels; rates are percentages.

    Do not merge label 0 into 2 before calling. Missing denominator classes
    yield 0.0. Legacy yellow transitions overlap with false alarms and
    unnecessary remeasurements; overall unsafe clearances aggregate 0/1/2 → 3.
    See docs/error_metrics.md for definitions and historical compatibility.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be one-dimensional and have equal lengths")

    # ============================================================
    # criticals: actual 1 or 2, predicted 3
    # ============================================================

    critical_mask = np.isin(y_true, [1, 2])
    critical_errors = np.sum(critical_mask & (y_pred == 3))

    critical_total = np.sum(critical_mask)

    critical_error_rate = (critical_errors / critical_total * 100
        if critical_total > 0
        else 0.0
    )

    # ============================================================
    # false alarms: actual 3, predicted 1 or 2
    # ============================================================

    false_alarms = np.sum((y_true == 3) & np.isin(y_pred, [1, 2]))

    false_alarm_total = np.sum(y_true == 3)

    false_alarm_rate = (false_alarms / false_alarm_total * 100
        if false_alarm_total > 0
        else 0.0
    )

    unsafe_clearances = np.sum((y_true == 0) & (y_pred == 3))
    poor_measurement_total = np.sum(y_true == 0)
    unsafe_clearance_rate = (
        unsafe_clearances / poor_measurement_total * 100
        if poor_measurement_total > 0 else 0.0
    )

    unnecessary_remeasurements = np.sum((y_true == 3) & (y_pred == 0))
    unnecessary_remeasurement_rate = (
        unnecessary_remeasurements / false_alarm_total * 100
        if false_alarm_total > 0 else 0.0
    )

    overall_unsafe_clearances = critical_errors + unsafe_clearances
    overall_unsafe_total = critical_total + poor_measurement_total
    overall_unsafe_clearance_rate = (
        overall_unsafe_clearances / overall_unsafe_total * 100
        if overall_unsafe_total > 0 else 0.0
    )

    # ============================================================
    # rot -> gelb: actual 1, predicted 0 oder 2
    # ============================================================

    red_to_yellow = np.sum((y_true == 1) & np.isin(y_pred, [0, 2]))

    red_total = np.sum(y_true == 1)
    red_to_yellow_rate = (red_to_yellow / red_total * 100
        if red_total > 0
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

        "unsafe_clearances": int(unsafe_clearances),
        "unsafe_clearance_rate": unsafe_clearance_rate,

        "unnecessary_remeasurements": int(unnecessary_remeasurements),
        "unnecessary_remeasurement_rate": unnecessary_remeasurement_rate,

        "overall_unsafe_clearances": int(overall_unsafe_clearances),
        "overall_unsafe_clearance_rate": overall_unsafe_clearance_rate,

        "red_to_yellow": int(red_to_yellow),
        "red_to_yellow_rate": red_to_yellow_rate,

        "green_to_yellow": int(green_to_yellow),
        "green_to_yellow_rate": green_to_yellow_rate,
    }
