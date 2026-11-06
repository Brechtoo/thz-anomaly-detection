import numpy as np
import pandas as pd
import pytest

from src.features.extract_tsfresh_features import extract_measurement, select_training_features


def test_extract_measurement_matches_signal_statistics(tmp_path):
    path = tmp_path / "signal.txt"
    np.savetxt(path, np.column_stack([np.arange(8), np.arange(8) ** 2]))
    result = extract_measurement((path, "minimal"))
    assert len(result) == 10
    assert result["y__maximum"] == 49
    assert result["y__mean"] == np.mean(np.arange(8) ** 2)


def test_empty_signal_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("# no values\n")
    with pytest.raises(ValueError, match="Empty or non-finite"):
        extract_measurement((path, "minimal"))


def test_selection_and_imputation_ignore_held_out_labels_and_values():
    labels = pd.Series(np.tile([0, 1, 2, 3], 100))
    train = labels.index[:320]
    features = pd.DataFrame({"signal": labels.astype(float), "constant": 1.0})
    baseline = select_training_features(features, labels, train)
    features.loc[320:, "signal"] = np.nan
    labels.loc[320:] = 0
    result = select_training_features(features, labels, train)
    assert list(result) == list(baseline) == ["signal"]
    pd.testing.assert_frame_equal(result.loc[train], baseline.loc[train])
    assert (result.loc[320:, "signal"] == 1.5).all()
