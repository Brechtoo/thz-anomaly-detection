"""Synthetic examples with hand-calculated expectations (no sklearn oracle)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_error_counts, evaluate_predictions


@pytest.mark.parametrize("container", [list, np.array, pd.Series])
def test_metrics_on_unbalanced_four_class_example(container):
    # Rows = truth, columns = prediction, ascending labels 0, 1, 2, 3.
    # Supports: 3, 2, 2, 3. Predicted totals: 4, 2, 2, 2.
    truth = container([0, 0, 0, 1, 1, 2, 2, 3, 3, 3])
    prediction = container([0, 0, 1, 1, 2, 2, 3, 0, 0, 3])
    result = evaluate_predictions(truth, prediction)

    assert set(result) == {
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
        "confusion_matrix", "classification_report",
    }
    assert result["accuracy"] == pytest.approx(5 / 10)
    assert result["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2 + 1 / 2 + 1 / 3) / 4)
    # F1 = 2 TP / (true support + predicted total).
    assert result["macro_f1"] == pytest.approx((4 / 7 + 1 / 2 + 1 / 2 + 2 / 5) / 4)
    assert result["weighted_f1"] == pytest.approx((3 * 4 / 7 + 2 * 1 / 2 + 2 * 1 / 2 + 3 * 2 / 5) / 10)
    np.testing.assert_array_equal(result["confusion_matrix"], [
        [2, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [2, 0, 0, 1],
    ])
    # Check formatted precision, recall, F1 and support independently.
    rows = [line.split() for line in result["classification_report"].splitlines()]
    class_rows = [row for row in rows if row and row[0] in {"0", "1", "2", "3"}]
    assert class_rows == [
        ["0", "0.5000", "0.6667", "0.5714", "3"],
        ["1", "0.5000", "0.5000", "0.5000", "2"],
        ["2", "0.5000", "0.5000", "0.5000", "2"],
        ["3", "0.5000", "0.3333", "0.4000", "3"],
    ]


@pytest.mark.parametrize("labels", [[0, 1, 2, 3], [0], [1], [2], [3]])
def test_perfect_predictions_including_missing_classes(labels):
    result = evaluate_predictions(labels, labels)
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        assert result[key] == pytest.approx(1.0)
    np.testing.assert_array_equal(result["confusion_matrix"], np.eye(len(labels), dtype=int))


def test_metrics_with_two_absent_classes():
    result = evaluate_predictions([1, 1, 3], [1, 3, 3])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["balanced_accuracy"] == pytest.approx((1 / 2 + 1) / 2)
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["weighted_f1"] == pytest.approx(2 / 3)
    np.testing.assert_array_equal(result["confusion_matrix"], [[1, 1], [0, 1]])


def test_class_present_only_in_predictions():
    # Recall for the absent true class is undefined; balanced accuracy uses
    # only true classes, while macro F1 includes the predicted-only class.
    with pytest.warns(UserWarning):
        result = evaluate_predictions([1, 1], [1, 3])
    assert result["accuracy"] == pytest.approx(1 / 2)
    assert result["balanced_accuracy"] == pytest.approx(1 / 2)
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["weighted_f1"] == pytest.approx(2 / 3)
    np.testing.assert_array_equal(result["confusion_matrix"], [[1, 1], [0, 0]])


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        evaluate_predictions([1, 2], [1])


# Explicit truth table covers every label pair and overlapping yellow counts.
@pytest.mark.parametrize("truth,prediction,critical,false_alarm,red_yellow,green_yellow,unsafe,remeasure", [
    (0, 0, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0),
    (0, 2, 0, 0, 0, 0, 0, 0), (0, 3, 0, 0, 0, 0, 1, 0),
    (1, 0, 0, 0, 1, 0, 0, 0), (1, 1, 0, 0, 0, 0, 0, 0),
    (1, 2, 0, 0, 1, 0, 0, 0), (1, 3, 1, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0, 0), (2, 1, 0, 0, 0, 0, 0, 0),
    (2, 2, 0, 0, 0, 0, 0, 0), (2, 3, 1, 0, 0, 0, 0, 0),
    (3, 0, 0, 0, 0, 1, 0, 1), (3, 1, 0, 1, 0, 0, 0, 0),
    (3, 2, 0, 1, 0, 1, 0, 0), (3, 3, 0, 0, 0, 0, 0, 0),
])
def test_error_counts_for_each_label_pair(truth, prediction, critical, false_alarm, red_yellow, green_yellow, unsafe, remeasure):
    result = calculate_error_counts([truth], [prediction])
    assert result == pytest.approx({
        "critical_errors": critical, "critical_error_rate": 100 * critical,
        "false_alarms": false_alarm, "false_alarm_rate": 100 * false_alarm,
        "unsafe_clearances": unsafe, "unsafe_clearance_rate": 100 * unsafe,
        "unnecessary_remeasurements": remeasure, "unnecessary_remeasurement_rate": 100 * remeasure,
        "overall_unsafe_clearances": critical + unsafe,
        "overall_unsafe_clearance_rate": 100 * (critical + unsafe),
        "red_to_yellow": red_yellow, "red_to_yellow_rate": 100 * red_yellow,
        "green_to_yellow": green_yellow, "green_to_yellow_rate": 100 * green_yellow,
    })


@pytest.mark.parametrize("container", [list, np.array, pd.Series])
def test_error_counts_and_percentage_denominators(container):
    # Three critical errors among five true labels 1/2; two false alarms
    # among four true labels 3. Label 0 contributes to neither denominator.
    result = calculate_error_counts(
        container([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
        container([3, 3, 1, 3, 3, 2, 1, 2, 0, 3]),
    )
    assert result == pytest.approx({
        "critical_errors": 3, "critical_error_rate": 60.0,
        "false_alarms": 2, "false_alarm_rate": 50.0,
        "unsafe_clearances": 1, "unsafe_clearance_rate": 100.0,
        "unnecessary_remeasurements": 1, "unnecessary_remeasurement_rate": 25.0,
        "overall_unsafe_clearances": 4, "overall_unsafe_clearance_rate": 400 / 6,
        "red_to_yellow": 0, "red_to_yellow_rate": 0.0,
        "green_to_yellow": 2, "green_to_yellow_rate": 50.0,
    })


def test_critical_rate_includes_correctly_predicted_label_two_in_denominator():
    result = calculate_error_counts([1, 2, 2], [3, 2, 2])
    assert result["critical_errors"] == 1
    assert result["critical_error_rate"] == pytest.approx(100 / 3)


def test_yellow_transition_rates_use_their_respective_true_class():
    result = calculate_error_counts(
        [1, 1, 1, 1, 2, 2, 3, 3, 3], [0, 2, 1, 1, 2, 2, 0, 2, 3],
    )
    assert result["red_to_yellow"] == 2
    assert result["red_to_yellow_rate"] == pytest.approx(50)
    assert result["green_to_yellow"] == 2
    assert result["green_to_yellow_rate"] == pytest.approx(200 / 3)
    assert result["critical_errors"] == 0
    assert result["false_alarms"] == 1
    assert result["unnecessary_remeasurements"] == 1


@pytest.mark.parametrize("truth,prediction", [
    ([0, 1, 2, 3], [0, 1, 2, 3]),
    ([0], [0]), ([1], [1]), ([2], [2]), ([3], [3]),
    # Incorrect predictions that belong to none of the defined error groups.
    ([0, 2], [1, 1]),
    ([], []),
])
def test_zero_error_counts_and_missing_denominators(truth, prediction):
    assert calculate_error_counts(truth, prediction) == {
        "critical_errors": 0, "critical_error_rate": 0.0,
        "false_alarms": 0, "false_alarm_rate": 0.0,
        "unsafe_clearances": 0, "unsafe_clearance_rate": 0.0,
        "unnecessary_remeasurements": 0, "unnecessary_remeasurement_rate": 0.0,
        "overall_unsafe_clearances": 0, "overall_unsafe_clearance_rate": 0.0,
        "red_to_yellow": 0, "red_to_yellow_rate": 0.0,
        "green_to_yellow": 0, "green_to_yellow_rate": 0.0,
    }


def test_overall_rate_is_weighted_by_support_not_mean_of_group_rates():
    result = calculate_error_counts([0, 0, 0, 1, 2, 3], [3, 0, 0, 3, 3, 3])
    assert result["unsafe_clearance_rate"] == pytest.approx(100 / 3)
    assert result["critical_error_rate"] == 100
    assert result["overall_unsafe_clearances"] == 3
    assert result["overall_unsafe_clearance_rate"] == 60


def test_original_labels_remain_distinct_and_inputs_are_not_modified():
    truth = pd.Series([0, 0, 1, 2, 3])
    prediction = np.array([3, 0, 3, 2, 0])
    truth_before = truth.copy()
    prediction_before = prediction.copy()
    # A separate RGY copy must never replace the original error inputs.
    rgy_truth = truth.replace({0: 2})
    result = calculate_error_counts(truth, prediction)
    assert result["unsafe_clearances"] == 1
    assert result["unsafe_clearance_rate"] == 50
    assert result["critical_errors"] == 1
    assert result["critical_error_rate"] == 50
    assert result["overall_unsafe_clearance_rate"] == 50
    remapped_result = calculate_error_counts(rgy_truth, prediction)
    assert remapped_result["unsafe_clearances"] == 0
    assert remapped_result["critical_errors"] == 2
    pd.testing.assert_series_equal(truth, truth_before)
    np.testing.assert_array_equal(prediction, prediction_before)


@pytest.mark.parametrize("truth,prediction", [
    ([0, 1], [3]), ([[0], [1]], [3, 3]), ([0, 1], [[3], [3]]),
])
def test_error_counts_reject_broadcastable_or_non_vector_inputs(truth, prediction):
    with pytest.raises(ValueError, match="one-dimensional.*equal lengths"):
        calculate_error_counts(truth, prediction)
