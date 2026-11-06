import numpy as np
import pandas as pd
import pytest

from src.evaluation.leakage import (
    group_overlap_table,
    parse_acquisition_metadata,
    spatial_train_proximity,
)


def test_group_overlap_distinguishes_shared_and_independent_sources():
    frame = pd.DataFrame({"source": ["a", "a", "b"], "split": ["train", "validation", "test"]})
    result = group_overlap_table(frame, ["source"]).set_index("source")
    assert result.loc["a", "train"] == result.loc["a", "validation"] == 1
    assert result.loc["a", "test"] == 0
    assert result.loc["a", "cross_split"]
    assert not result.loc["b", "cross_split"]
    assert result.loc["b", "split_count"] == 1


@pytest.mark.parametrize("source,split", [(None, "train"), ("a", None), ("a", "unknown")])
def test_overlap_rejects_missing_or_unknown_metadata(source, split):
    with pytest.raises(ValueError):
        group_overlap_table(pd.DataFrame({"source": [source], "split": [split]}), ["source"])


def test_filename_metadata_is_explicit_not_a_guessed_session():
    names = pd.Series(["2026-04-27T16-02-34.891893-Hohlrmtestk1_tr-[117]-[-79.0,-32.0,3.59]-[1.0,0.0,0.0,0.0]-delta[0.013mm-0.0deg]-avg20.txt"])
    metadata = parse_acquisition_metadata(names)
    assert metadata.scan_name.iloc[0] == "Hohlrmtestk1_tr"
    assert metadata.sequence.iloc[0] == 117
    assert metadata.acquisition_date.iloc[0] == "2026-04-27"
    assert metadata.average.iloc[0] == 20
    with pytest.raises(ValueError):
        parse_acquisition_metadata(pd.Series(["unrecognized.txt"]))


def test_nearest_train_is_restricted_to_same_source_and_does_not_change_splits():
    frame = pd.DataFrame({
        "source_dataset": ["a", "a", "a", "b"],
        "instance_id": ["a0", "a1", "a2", "b0"],
        "split": ["train", "validation", "test", "validation"],
        "x": [0, 1, 3, 0], "y": [0, 0, 4, 0],
    })
    before = frame.copy(deep=True)
    result = spatial_train_proximity(frame).set_index("instance_id")
    assert result.loc["a1", "nearest_train_xy_distance"] == 1
    assert result.loc["a2", "nearest_train_xy_distance"] == 5
    assert np.isinf(result.loc["b0", "nearest_train_xy_distance"])
    pd.testing.assert_frame_equal(frame, before)
