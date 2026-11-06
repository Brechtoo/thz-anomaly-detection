import json

import numpy as np
import pandas as pd
import pytest

from src.features import rebuild_feature_table as rebuild
from src.features.extract_all_features import extract_all_features
from src.features.extract_baseline_features import extract_baseline_features


@pytest.fixture
def signal(tmp_path):
    x = np.linspace(970, 1085, 2301)
    y = -10 * np.exp(-((x - 990) / 0.3) ** 2) + np.sin(x * 2) * 0.1
    path = tmp_path / "signal.txt"
    np.savetxt(path, np.column_stack([x, y]))
    return path


def test_baseline_area_matches_hand_calculated_trapezoids(tmp_path):
    path = tmp_path / "simple.txt"
    path.write_text("# x y\n0 1\n1 -2\n2 3\n")
    result = extract_baseline_features(path)
    assert result["y_abs_area"] == 4.0  # (1+2)/2 + (2+3)/2
    assert result["y_energy"] == 14.0


def test_all_52_features_are_finite_and_repeatable(signal):
    first = extract_all_features(signal)
    second = extract_all_features(signal)
    assert len(first) == 52
    assert first == second
    assert all(np.isfinite(value) for value in first.values())


def test_safe_rebuild_preserves_order_and_records_input_hashes(signal, tmp_path, monkeypatch):
    monkeypatch.setattr(rebuild, "check_locked_environment", lambda path: None)
    monkeypatch.setattr(rebuild, "capture_git_state", lambda root: {"git_commit": "test", "git_dirty": False})
    data = pd.DataFrame({
        "file_path": [str(signal), str(signal)], "instance_id": ["b", "a"],
        "label": [2, 0], "source_dataset": ["synthetic"] * 2,
        "x": [0, 1], "y": [0, 1],
    })
    input_path = tmp_path / "input.csv"
    data.to_csv(input_path, index=False)
    before = input_path.read_bytes(), signal.read_bytes()
    output = tmp_path / "new"
    path = rebuild.rebuild_feature_table(input_path, output)
    result = pd.read_csv(path)
    assert result.instance_id.tolist() == ["b", "a"]
    assert result.label.tolist() == [2, 0]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["row_count"] == 2
    assert len(manifest["feature_columns"]) == 52
    assert manifest["input_csv_sha256"] == rebuild.sha256_file(input_path)
    assert manifest["output_csv_sha256"] == rebuild.sha256_file(path)
    assert len(pd.read_csv(output / "raw_inputs.csv")) == 2
    assert before == (input_path.read_bytes(), signal.read_bytes())
    with pytest.raises(FileExistsError):
        rebuild.rebuild_feature_table(input_path, output)
    assert path.read_bytes()  # prior output survives refusal


def test_rebuild_requires_both_relocation_roots(tmp_path):
    with pytest.raises(ValueError, match="supplied together"):
        rebuild.rebuild_feature_table(tmp_path / "input.csv", tmp_path / "new", raw_root=tmp_path)
